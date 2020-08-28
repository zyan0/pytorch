import argparse
import atexit
import datetime
import multiprocessing
import multiprocessing.dummy
import os
import pickle
import queue
import subprocess
import tempfile
import threading
import time

import torch
from torch.utils._benchmark import Timer, Compare

_MIN_RUN_TIME = 1
_RESULT_FILE = "/tmp/blas_results.pkl"
_RESULT_FILE_LOCK = threading.Lock()

_WORKER_POOL = queue.Queue()
def clear_worker_pool():
    while not _WORKER_POOL.empty():
        _, result_file, _ = _WORKER_POOL.get_nowait()
        os.remove(result_file)
atexit.register(clear_worker_pool)


_BLAS_CONFIGS = (
    ("MKL (2020.2)", "blas_compare_mkl_2020_2", None),
    ("MKL (2020.1)", "blas_compare_mkl_2020_1", None),
    ("MKL (2020.1), MKL_DEBUG_CPU_TYPE=5", "blas_compare_mkl_2020_1", {"MKL_DEBUG_CPU_TYPE": "5"}),
    ("MKL (2020.0)", "blas_compare_mkl_2020_0", None),
    ("MKL (2020.0), MKL_DEBUG_CPU_TYPE=5", "blas_compare_mkl_2020_0", {"MKL_DEBUG_CPU_TYPE": "5"}),
    ("OpenBLAS", "blas_compare_openblas", None),
    # ("BLIS", "blas_compare_blis", None),
    # ("Eigen", "blas_compare_eigen", None),
)

_EXCLUDE_LAPACK = ("BLIS", "Eigen")


def fill_core_pool(n: int):
    clear_worker_pool()

    # Reserve two cores so that bookkeeping does not interfere with runs.
    cpu_count = multiprocessing.cpu_count() - 2

    # Adjacent cores sometimes share cache, so we space out single core runs.
    step = max(n, 2)
    for i in range(0, cpu_count, step):
        core_str = f"{i}" if n == 1 else f"{i},{i + n - 1}"
        _, result_file = tempfile.mkstemp(suffix=".pkl")
        _WORKER_POOL.put((core_str, result_file, n))


def _subprocess_main(seed=0, num_threads=1, sub_label="N/A", result_file=None, env=None):
    torch.manual_seed(seed)
    results = []
    for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 7, 96, 150, 225]:
        shapes = [
            ((n, n), (n, n), "(n x n) x (n x n)"),
            ((16, n), (n, n), "(16 x n) x (n x n)"),
            ((n, n), (n, 16), "(n x n) x (n x 16)")
        ]
        for x_shape, y_shape, shape_str in shapes:
            t = Timer(
                stmt="torch.mm(x, y)",
                label=f"torch.mm {shape_str}",
                sub_label=sub_label,
                description=f"n = {n}",
                env=env,
                globals={
                    "x": torch.rand(x_shape),
                    "y": torch.rand(y_shape),
                },
                num_threads=num_threads,
            ).blocked_autorange(min_run_time=_MIN_RUN_TIME)
            results.append(t)

        if sub_label not in _EXCLUDE_LAPACK:
            t = Timer(
                stmt="torch.eig(x)",
                label=f"torch.eig",
                sub_label=sub_label,
                description=f"n = {n}",
                env=env,
                globals={
                    "x": torch.rand((n, n)),
                },
                num_threads=num_threads,
            ).blocked_autorange(min_run_time=_MIN_RUN_TIME)
            results.append(t)

            # Sample until the Cholesky decomposition is non-singular.
            for _ in range(100):
                try:
                    x_base = torch.rand((n, n)) / n ** 0.5
                    x = torch.mm(x_base, x_base.t()) + 1e-2
                    torch.cholesky(x)
                    break
                except RuntimeError:
                    continue

            t = Timer(
                stmt="torch.cholesky(x)",
                label=f"torch.cholesky",
                sub_label=sub_label,
                description=f"n = {n}",
                env=env,
                globals={
                    "x": x,
                },
                num_threads=num_threads,
            ).blocked_autorange(min_run_time=_MIN_RUN_TIME)
            results.append(t)

            shapes = [
                ((n, n), "(n x n)"),
                ((16, n), "(16 x n)"),
                ((n, 16), "(n x 16)")
            ]
            for x_shape, shape_str in shapes:
                t = Timer(
                    stmt="torch.svd(x)",
                    label=f"torch.svd {shape_str}",
                    sub_label=sub_label,
                    description=f"n = {n}",
                    env=env,
                    globals={
                        "x": torch.rand(x_shape),
                    },
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=_MIN_RUN_TIME)
                results.append(t)

    if result_file is not None:
        with open(result_file, "wb") as f:
            pickle.dump(results, f)


def run_subprocess(args):
    seed, env, sub_label, extra_env_vars = args
    core_str = None
    try:
        core_str, result_file, num_threads = _WORKER_POOL.get()
        with open(result_file, "wb"):
            pass

        env_vars = {
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH") or "",

            # NumPy
            "OMP_NUM_THREADS": str(num_threads),
            "MKL_NUM_THREADS": str(num_threads),
            "NUMEXPR_NUM_THREADS": str(num_threads),
        }
        env_vars.update(extra_env_vars or {})

        subprocess.run(
            f"source activate {env} && "
            f"taskset --cpu-list {core_str} "
            "python blas_compare.py "
            "--DETAIL_in_subprocess "
            f"--DETAIL_seed {seed} "
            f"--DETAIL_num_threads {num_threads} "
            f"--DETAIL_sub_label '{sub_label}' "
            f"--DETAIL_result_file {result_file} "
            f"--DETAIL_env {env}",
            env=env_vars,
            stdout=subprocess.PIPE,
            shell=True
        )

        with open(result_file, "rb") as f:
            result_bytes = f.read()

        with _RESULT_FILE_LOCK, \
             open(_RESULT_FILE, "ab") as f:
            f.write(result_bytes)

    except KeyboardInterrupt:
        pass  # Handle ctrl-c gracefully.

    finally:
        if core_str is not None:
            _WORKER_POOL.put((core_str, result_file, num_threads))


def main():
    with open(_RESULT_FILE, "wb"):
        pass

    for num_threads in [1, 2, 4]:
        fill_core_pool(num_threads)
        workers = _WORKER_POOL.qsize()

        trials = []
        for seed in range(10):
            for sub_label, env, extra_env_vars in _BLAS_CONFIGS:
                trials.append((seed, env, sub_label, extra_env_vars))

        n = len(trials)
        with multiprocessing.dummy.Pool(workers) as pool:
            start_time = time.time()
            for i, r in enumerate(pool.imap(run_subprocess, trials)):
                n_trials_done = i + 1
                time_per_result = (time.time() - start_time) / n_trials_done
                eta = int((n - n_trials_done) * time_per_result)
                print(f"\r{i + 1} / {n}    ETA:{datetime.timedelta(seconds=eta)}".ljust(80), end="")
        print()

    with open(_RESULT_FILE, "rb") as f:
        results = []
        while True:
            try:
                results.extend(pickle.load(f))
            except EOFError:
                break

    comparison = Compare(results)
    comparison.trim_significant_figures()
    comparison.colorize()
    comparison.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DETAIL_in_subprocess", action="store_true")
    parser.add_argument("--DETAIL_seed", type=int, default=None)
    parser.add_argument("--DETAIL_num_threads", type=int, default=None)
    parser.add_argument("--DETAIL_sub_label", type=str, default="N/A")
    parser.add_argument("--DETAIL_result_file", type=str, default=None)
    parser.add_argument("--DETAIL_env", type=str, default=None)
    args = parser.parse_args()

    if args.DETAIL_in_subprocess:
        try:
            _subprocess_main(args.DETAIL_seed, args.DETAIL_num_threads, args.DETAIL_sub_label, args.DETAIL_result_file, args.DETAIL_env)
        except KeyboardInterrupt:
            pass  # Handle ctrl-c gracefully.
    else:
        main()
