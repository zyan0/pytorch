import argparse
import enum
import multiprocessing
import multiprocessing.dummy
import os
import pickle
import tempfile
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing
from torch.utils.benchmark import Language, Measurement, Timer

import tasks

NUM_CORES = multiprocessing.cpu_count()
NUM_WORKERS = max(NUM_CORES - 4, 1)
CALLGRIND_NUMBER =  10_000
MIN_RUN_TIME = 5


class Measured(enum.Enum):
    CALLGRIND = 0
    WALL_TIME = 1


def measure_python_times(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--sleep", type=int, default=0)
    argv = parser.parse_args(argv)

    time.sleep(argv.sleep)

    with open(argv.input, "rb") as f:
        map_args: List[Tuple[Tuple[str, ...], tasks.Mode, tasks.TimerSpec]] = pickle.load(f)

    results = []
    for label, mode, timer_spec in map_args:
        timer = tasks.make_timer(timer_spec)
        results.append((label, mode, timer.blocked_autorange(min_run_time=MIN_RUN_TIME)))

    with open(argv.output, "wb") as f:
        pickle.dump(results, f)


def map_fn(args):
    label, mode, measured, timer = args

    if measured == Measured.WALL_TIME:
        result = timer.blocked_autorange(min_run_time=MIN_RUN_TIME)
    else:
        result = timer.collect_callgrind(
            number=CALLGRIND_NUMBER,
            collect_baseline=False
        )

    return label, mode, measured, result


def collect():
    benchmark_tasks = tasks.Tasks()
    labels, map_args, side_map_args = [], [], []
    for label, task in benchmark_tasks.items():
        labels.append(label)
        for mode, task_spec in task.sub_tasks.items():
            timer = tasks.make_timer(task_spec)
            map_args.append((label, mode, Measured.CALLGRIND, timer))
            if task_spec.language == Language.PYTHON:
                side_map_args.append((label, mode, task_spec))
            else:
                map_args.append((label, mode, Measured.WALL_TIME, timer))

    # This reduces overall time by limiting straggler effects.
    map_args.sort(key=lambda x: (
        x[2] == Measured.CALLGRIND,
        "backward" in x[3]._task_spec.stmt,
        x[1] in (tasks.Mode.CPP, tasks.Mode.CPP_TS),
    ), reverse=True)

    _, side_map_input = tempfile.mkstemp(suffix=".pkl")
    _, side_map_output = tempfile.mkstemp(suffix=".pkl")
    python_timing_proc = None
    try:
        with open(side_map_input, "wb") as f:
            pickle.dump(side_map_args, f)

        python_timing_proc = subprocess.Popen(
            [
                sys.executable, os.path.abspath(__file__),
                "--mode", "DETAIL_measure_python_times",
                "--input", side_map_input,
                "--output", side_map_output,
                "--sleep", "60",  # Don't contend with the initial compilations.
            ],
            shell=False,
        )

        results = {Measured.WALL_TIME: {}, Measured.CALLGRIND: {}}
        with multiprocessing.dummy.Pool(NUM_WORKERS) as pool:
            start_time = time.time()
            try:
                for i, (label, mode, measured, result) in enumerate(pool.imap_unordered(map_fn, map_args, 1)):
                    results[measured][(label, mode)] = result
                    print(f"\r{i + 1} / {len(map_args)}", end="")
                    sys.stdout.flush()
                print(f"Total time: {time.time() - start_time:.0f} seconds")
            except KeyboardInterrupt:
                outstanding = "\n".join([
                    f"{'_'.join(label)}  {mode}  {measured}"
                    for label, mode, measured, _ in map_args
                    if (label, mode) not in results[measured]
                ])
                raise KeyboardInterrupt(f"Outstanding runs:\n{outstanding}")

        if not python_timing_proc.poll() is None:
            print("Waiting for Python times.")
            python_timing_proc.wait()

        with open(side_map_output, "rb") as f:
            python_times: List[Tuple[Tuple[str, ...], tasks.Mode, Measurement]] = pickle.load(f)

        for label, mode, result in python_times:
            results[Measured.WALL_TIME][(label, mode)] = result

    finally:
        if python_timing_proc is not None:
            python_timing_proc.kill()
        os.remove(side_map_input)
        os.remove(side_map_output)

    import pdb
    pdb.set_trace()
    return labels, results


def main(argv):
    if argv:
        raise ValueError(f"`main` does not support arguments. Got: {argv}")


    labels, results = collect()

    for label in labels:
        print(", ".join(label))
        for mode in (tasks.Mode.PY, tasks.Mode.PY_TS, tasks.Mode.CPP, tasks.Mode.CPP_TS):
            k = (label, mode)
            if k in results[Measured.CALLGRIND]:
                c = results[Measured.CALLGRIND][k].counts(denoise=True)
                t = results[Measured.WALL_TIME][k].median
                print(f"  {mode:<15}  {c:>12}  {t * 1e6:>8.1f} us")
        print()


_MODES = {
    "main": main,
    "DETAIL_measure_python_times": measure_python_times,
}


if __name__ == "__main__":
    raise NotImplementedError("Currently transitioning.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=list(_MODES.keys()), default="main")
    args, unknown_args = parser.parse_known_args()
    _MODES[args.mode](unknown_args)
