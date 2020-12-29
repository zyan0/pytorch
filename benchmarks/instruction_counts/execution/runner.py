"""Orchestrates benchmark collection across many cores."""
import statistics
import subprocess
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING

from core.api import WorkerFailed
from execution.cores import CorePool, CPU_COUNT, SLACK
from execution.future import WorkFuture
from execution.worker import MIN_RUN_TIME

if TYPE_CHECKING:
    # See core.api for an explanation.
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language


class Runner:
    _core_pool: CorePool
    _work_items: Tuple[WorkFuture, ...]
    _start_time: Optional[float]
    _job_queue: List[WorkFuture]
    _active_jobs: List[WorkFuture]
    _core_allocation: Dict[WorkFuture, str]
    _finished_jobs: List[WorkFuture]

    def __init__(self, work_items: Tuple[WorkFuture, ...]) -> None:
        self._core_pool = CorePool()
        self._work_items = work_items
        self._start_time = None
        self._job_queue = list(work_items)
        self._active_jobs = []
        self._core_allocation = {}
        self._finished_jobs = []

        if len(work_items) != len(set(work_items)):
            raise ValueError('Duplicate work items.')

        if any(w.started for w in work_items):
            raise ValueError('Work items must cannot already be started.')

    def run(self) -> None:
        try:
            self._run()

        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt (ctrl-c) detected. Shutting down children.")
            self._force_shutdown()
            raise

        except subprocess.TimeoutExpired:
            print("\n\nJob times out. Shutting down children.")
            self._force_shutdown()
            raise

        except WorkerFailed as e:
            print(f'\n\nWorker failed: {e.timer_args}')
            print('Shutting down all outstanding jobs before re-raising.')
            self._force_shutdown()
            if e.wrapped_trace:
                print(e.wrapped_trace)
            else:
                print('Unknown failure. (Worker did not report exception contents.)')
            raise

        except:
            print("\n\nUnknown exception. Shutting down jobs before re-raising.")
            self._force_shutdown()
            raise

    def _run(self) -> None:
        self._start_time = time.time()
        self._canary_import()
        while self._job_queue or self._active_jobs:
            t0 = time.time()
            self._update_active_jobs()
            self._enqueue_new_jobs()
            self._display_progress()
            time.sleep(1.0 - (time.time() - t0))
        print(f"\nTotal time: {time.time() - self._start_time:.0f} seconds")

    def _update_active_jobs(self) -> None:
        active_jobs: List[WorkFuture] = []
        for job in self._active_jobs:
            if not job.ready:
                active_jobs.append(job)

            elif job.result is not None:
                self._finished_jobs.append(job)
                self._core_pool.release(self._core_allocation[job])

            else:
                assert job.worker_failure is not None
                raise WorkerFailed(
                    timer_args=job._timer_args,
                    wrapped_trace=job.worker_failure.failure_trace,
                )
        self._active_jobs.clear()
        self._active_jobs.extend(active_jobs)

    def _enqueue_new_jobs(self) -> None:
        job_queue: List[WorkFuture] = []
        for job in self._job_queue:
            cpu_list: Optional[str] = self._core_pool.reserve(job.num_cores)
            if cpu_list is None:
                job_queue.append(job)
            else:
                job.start(cpu_list=cpu_list)
                self._core_allocation[job] = cpu_list
                self._active_jobs.append(job)
        self._job_queue.clear()
        self._job_queue.extend(job_queue)

    @staticmethod
    def group_by_language(
        items: Iterable[WorkFuture]
    ) -> Dict[Language, float]:
        grouped: Dict[Language, List[WorkFuture]] = {}
        for w in items:
            grouped.setdefault(w.language, [])
            grouped[w.language].append(w)

        return {
            k: statistics.mean((w.run_time for w in v))
            for k, v in grouped.items()
        }

    def _display_progress(self) -> None:
        now = time.time()

        assert self._start_time is not None
        elapsed = now - self._start_time

        approximate_estimate: bool = False
        time_estimates = self.group_by_language(self._finished_jobs)
        cpu_time_estimates: List[Tuple[int, float]] = []
        for w in self._active_jobs + self._job_queue:
            if w.language in time_estimates:
                time_estimate = time_estimates[w.language]
            else:
                approximate_estimate = True
                time_estimate = (
                    MIN_RUN_TIME +

                    # Callgrind takes about just under minute.
                    50.0 +

                    # C++ compilation takes about 20 seconds, and there are two
                    # of them. (One for wall time and one for callgrind.)
                    (2 * 20.0 if w.language == Language.CPP else 0.0)
                )

            if w.started:
                # Factor in elapsed time.
                time_estimate = max(time_estimate - elapsed, 0)
            cpu_time_estimates.append((w.num_cores, time_estimate))

        # Assume ideal core utilization.
        overall_remaining = sum(c * t for c, t in cpu_time_estimates) / (CPU_COUNT - SLACK)

        # If the time remaining is < 10 minutes, switch to a more precise
        # bin-packing scheme which will better predict straggler effects.
        # This isn't a particularly efficient algorithm and it's not EXACTLY
        # what CorePool, but it's good enough for an estimate. (And it's not
        # on the hot path.)
        if overall_remaining < 600:
            core_times = [0.0 for _ in range(CPU_COUNT - SLACK)]
            for num_cores, time_estimate in cpu_time_estimates:
                for i in range(num_cores):
                    core_times[i] = core_times[i + num_cores - 1] + time_estimate
                core_times.sort()
            overall_remaining = max(core_times)

        if not overall_remaining:
            eta_str = f"ETA: Soon"
        else:
            eta_str = (
                f"ETA{' (approximate)' if approximate_estimate else ''}: "
                f"{overall_remaining:.0f} seconds")

        core_seconds_used = (
            sum((w.run_time * w.num_cores) for w in self._finished_jobs) +
            sum(now - w.start_time for w in self._active_jobs))
        packing_efficiency = core_seconds_used / (elapsed * (CPU_COUNT - SLACK))

        print(
            f"\r{len(self._finished_jobs)} / {len(self._work_items)} "
            f"{eta_str}, Job packing efficiency: {packing_efficiency * 100:.1f}%".ljust(80),
            end="",
        )

    def _force_shutdown(self) -> None:
        """Try to interrupt jobs, and kill if need be.

        We would prefer to softly terminate jobs so that they have a chance to
        clean up before shutting down.
        """
        for job in self._active_jobs:
            job.interrupt()

        if self._active_jobs:
            time.sleep(0.5)

        remaining_jobs: List[WorkFuture] = [j for j in self._active_jobs if not j.ready]
        if remaining_jobs:
            print(
                f'SIGINT sent to {len(self._active_jobs)} jobs, '
                f'{len(remaining_jobs)} have not yet exited.\n'
                'Entering short cleanup loop, after which stragglers will '
                'be forcibly terminated.'
            )

            for _ in range(5):
                time.sleep(1.0)
                remaining_jobs = [j for j in remaining_jobs if not j.ready]
                if remaining_jobs:
                    print(f'{len(remaining_jobs)} still remain.')
                else:
                    print('All remaining jobs have gracefully terminated.')
                    return

            print(f'{len(remaining_jobs)} jobs refused to exit. Forcibly terminating.')
            for j in remaining_jobs:
                j.terminate()

    def _canary_import(self) -> None:
        """Make sure we can import torch before launching a slew of workers."""
        source_cmds: Set[str] = set()
        for w in self._work_items:
            if w._source_cmd is not None:
                source_cmds.add(f"{w._source_cmd} && ")

        for source_cmd in (source_cmds or {""}):
            cmd = f'{source_cmd}python -c "import torch"'
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
            )

            if proc.returncode:
                raise ImportError(
                    f'Failed to import torch in subprocess: {cmd}\n{proc.stdout}')
