import atexit
import os
import pickle
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

import runner.core

CALLGRIND = runner.core.MeasurementType.CALLGRIND

class _TempfilePool:
    def __init__(self):
        all_files = []
        def cleanup():
            for fpath in all_files:
                try:
                    os.remove(fpath)
                except:
                    print(f'Failed to clean up {fpath}')
        atexit.register(cleanup)

        self._lock = threading.Lock()
        self._all_files = all_files
        self._available_files = []

    def get(self):
        with self._lock:
            if self._available_files:
                fpath = self._available_files.pop()
            else:
                _, fpath = tempfile.mkstemp(suffix='.pkl')
                self._all_files.append(fpath)
        return fpath

    def release(self, fpath: str):
        with open(fpath, 'wb') as f:
            # Ensure file is empty.
            pass

        with self._lock:
            self._available_files.append(fpath)

_TEMPFILE_POOL = _TempfilePool()


class WorkerFailed(Exception):
    def __init__(
        self,
        worker_input: runner.core.WorkerInput,
        e_str: Optional[str] = None
    ) -> None:
        self.worker_input: runner.core.WorkerInput = worker_input
        self.wrapped_exception_str: Optional[Exception] = e_str
        super().__init__()


class WorkItem:
    def __init__(self, worker_input: runner.core.WorkerInput, source_cmd=None):
        self._lock = threading.RLock()
        self._worker_input: core.WorkerInput = worker_input
        self._source_cmd: Optional[str] = source_cmd

        self._started = False
        self._finished = False
        self._start_time = None
        self._communication_file: Optional[str] = None
        self._proc: Optional[subprocess.CompletedProcess] = None
        self._worker_output: Optional[core.WorkerOutput] = None

    def start(self, cpu_list: Optional[str] = None):
        with self._lock:
            assert not self._started, 'Already started.'
            self._communication_file = _TEMPFILE_POOL.get()
            with open(self._communication_file, 'wb') as f:
                pickle.dump(self._worker_input, f)

            cmd = ' '.join([
                'python', '-m', 'runner.worker',
                '--communication_file', self._communication_file,
            ])
            if cpu_list is not None:
                cmd = f'taskset --cpu-list {cpu_list} {cmd}'
            if self._source_cmd is not None:
                cmd = f'{source_cmd} && {cmd}'

            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
            )
            self._started = True

    def interrupt(self):
        if self._proc is not None:
            self._proc.send_signal(signal.SIGINT)

    def terminate(self):
        if self._proc is not None:
            self._proc.terminate()

    @property
    def started(self):
        return self._started

    @property
    def returncode(self):
        with self._lock:
            return self._proc.poll() if self._started else None

    @property
    def measurement_type(self):
        return self._worker_input.measurement_type

    @property
    def result(self):
        if self._worker_output:
            return self._worker_output

        with self._lock:
            self._lazy_collect()
            return self._worker_output

    def _lazy_collect(self):
        if self._worker_output is not None:
            return

        with self._lock:
            if self._communication_file is not None:
                if self.returncode is 0:
                    with open(self._communication_file, 'rb') as f:
                        self._worker_output = pickle.load(f)
                    assert isinstance(self._worker_output, runner.core.WorkerOutput)

                _TEMPFILE_POOL.release(self._communication_file)
                self._communication_file = None
                self._finished = True

    def __hash__(self):
        return hash(id(self))


class Runner:
    def __init__(self, work_items: Tuple[WorkItem, ...]):
        self._work_items: Tuple[WorkItem, ...] = work_items
        if len(work_items) != len(set(work_items)):
            raise ValueError('Duplicate work items.')

        if any(w.started for w in work_items):
            raise ValueError('Work items must cannot already be started.')

        self._num_work_items_by_type: Dict[runner.core.MeasurementType, int] = {
            measurement_type: sum(w.measurement_type == measurement_type for w in work_items)
            for measurement_type in runner.core.MeasurementType
        }

        self._active_jobs: Dict[str, Optional[WorkItem]] = {}

    def run(self):
        try:
            self._run()

        except KeyboardInterrupt:
            print('\n\nKeyboardInterrupt (ctrl-c) detected. Shutting down children.')
            self.force_shutdown()
            raise

        except WorkerFailed as e:
            print(f'\n\nWorker failed: {e.worker_input}')
            print('Shutting down all outstanding jobs before re-raising.')
            self.force_shutdown()
            if e.wrapped_exception_str:
                print(e.wrapped_exception_str)
            else:
                print('Unknown failure. (Worker did not report exception contents.)')
            raise

        except:
            print('\n\nUnknown exception. Shutting down jobs before re-raising.')
            self.force_shutdown()
            raise

    def _run(self):
        self.canary_import()

        work_item_queue: List[WorkItem] = list(self._work_items[::-1])
        self._active_jobs = {
            str(i): None
            for i in range(runner.core.NUM_WORKERS)
        }

        start_time: float = time.time()
        item_start_times: Dict[WorkItem, float] = {}
        item_end_times: Dict[WorkItem, float] = {}
        finished_work_items: List[WorkItem] = []
        while self._active_jobs:
            for cpu_core in list(self._active_jobs.keys()):
                j = self._active_jobs[cpu_core]
                if j is not None:
                    retcode = j.returncode
                    if retcode is None:
                        # Still running
                        pass

                    elif retcode is not 0:
                        # We failed in a way that the worker couldn't catch.
                        # (e.g. couldn't unpickle inputs, failed import, etc.)
                        raise WorkerFailed(j._worker_input)

                    else:
                        if j.result.e_str:
                            # We failed inside the worker's try-catch block.
                            # (e.g. bad stmt, missing Valgrind, etc.)
                            raise WorkerFailed(j._worker_input, j.result.e_str)

                        finished_work_items.append(j)
                        item_end_times[j] = time.time()
                        self._active_jobs[cpu_core] = j = None

                if j is None:
                    if work_item_queue:
                        self._active_jobs[cpu_core] = j = work_item_queue.pop()
                        j.start(cpu_list=cpu_core)
                        item_start_times[j] = time.time()
                    else:
                        self._active_jobs.pop(cpu_core)

            self.print_eta_estimate(
                finished_work_items,
                start_time,
                item_start_times,
                item_end_times,
            )
            time.sleep(2)

        import pdb
        pdb.set_trace()

    def canary_import(self):
        # Make sure we can import torch before launching a bunch of workers.
        source_cmds = set()
        for w in self._work_items:
            if w._source_cmd is not None:
                source_cmds.add(w._source_cmd)

        if source_cmds:
            cmds = [
                f'{source_cmd} && python -c "import torch"'
                for source_cmd in source_cmds
            ]
        else:
            cmds = ['python -c "import torch"']

        for cmd in cmds:
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if proc.returncode:
                print(proc.stdout.decode('utf-8'))
                raise ImportError(f'Failed to import torch in subprocess: {cmd}')

    def print_eta_estimate(
        self,
        finished_work_items: List[WorkItem],
        start_time: float,
        item_start_times: Dict[WorkItem, float],
        item_end_times: Dict[WorkItem, float],
    ) -> None:
        if len(finished_work_items) == len(self._work_items):
            print(f'\rFinished. Total time: {time.time() - start_time:.0f} sec')

        finished_by_type: Dict[runner.core.MeasurementType, List[WorkItem]] = {
            measurement_type: [
                w for w in finished_work_items
                if w.measurement_type == measurement_type
            ]
            for measurement_type in runner.core.MeasurementType
        }

        approx_t_remaining = 0
        for measurement_type, n in self._num_work_items_by_type.items():
            finished = finished_by_type[measurement_type]
            if n and not finished:
                print('\rETA: unknown', end='')
                return

            time_per_item = statistics.mean(
                (item_end_times[w] - item_start_times[w] for w in finished))

            approx_t_remaining += time_per_item * (n - len(finished))

        now = time.time()
        for w in self._active_jobs.values():
            assert w is not None
            approx_t_remaining -= (now - item_start_times[w])

        if approx_t_remaining <= 0:
            print(f'\rETA: soon ({len(self._active_jobs)} stragglers)', end='')
        else:
            print(f'\rETA: {approx_t_remaining:.0f} sec', end='')

    def force_shutdown(self):
        if not self._active_jobs:
            return

        for work_item in self._active_jobs.values():
            work_item.interrupt()

        remaining_jobs = [
            w for w in self._active_jobs.values()
            if w.started and w.returncode is None
        ]

        if remaining_jobs:
            print(
                f'SIGINT sent to {len(self._active_jobs)} jobs, '
                f'{len(remaining_jobs)} have not yet exited.\n'
                'Entering short cleanup loop, after which stragglers will '
                'be forcibly terminated.'
            )
            for _ in range(5):
                time.sleep(1.0)
                remaining_jobs = [r for r in remaining_jobs if r.returncode is None]
                if remaining_jobs:
                    print(f'{len(remaining_jobs)} still remain.')
                else:
                    print('All remaining jobs have gracefully terminated.')
                    return

            print(f'{len(remaining_jobs)} jobs refused to exit. Forcibly terminating.')
            for w in remaining_jobs:
                w.terminate()
