import json
import os
import pickle
import signal
import subprocess
import time
from typing import List, Optional, Union, TYPE_CHECKING
import uuid

from core.api import Mode, TimerArgs, WorkerFailure, WorkerOutput
from core.types import Label
from core.utils import get_temp_dir

if TYPE_CHECKING:
    # See core.api for an explanation of Language import.
    from torch.utils.benchmark.utils.timer import Language
    PopenType = subprocess.Popen[bytes]
else:
    from torch.utils.benchmark import Language
    PopenType = subprocess.Popen


class WorkFuture:
    """Wraps a call to `worker.py` in a future API.

    NB: This object is NOT thread safe.
    """

    # Constructor arguments.
    _timer_args: TimerArgs
    _source_cmd: Optional[str]
    _timeout: Optional[float]
    _label: Optional[Label]
    _mode: Optional[Mode]

    # Internal bookkeeping
    _start_time: Optional[float] = None
    _end_time: Optional[float] = None
    _cmd: Optional[str] = None
    _returncode: Optional[int] = None
    _communication_file: Optional[str] = None
    _proc: Optional[PopenType] = None
    _output: Optional[WorkerOutput] = None
    _worker_failure: Optional[WorkerFailure] = None

    def __init__(
        self,
        timer_args: TimerArgs,
        source_cmd: Optional[str] = None,
        timeout: Optional[float] = None,
        label: Optional[Label] = None,
        mode: Optional[Mode] = None,
    ) -> None:
        self._timer_args = timer_args
        self._source_cmd = source_cmd
        self._timeout = timeout

        # For failure diagnostics.
        self._label = label
        self._mode = mode

    def __hash__(self) -> int:
        return hash(id(self))

    @property
    def num_cores(self) -> int:
        assert isinstance(self._timer_args.num_threads, int)
        return self._timer_args.num_threads

    @property
    def started(self) -> bool:
        return self._start_time is not None

    @property
    def ready(self) -> bool:
        return self.returncode is not None

    @property
    def returncode(self) -> Optional[int]:
        # Process has either not started or we've alredy determined the returncode.
        if not self.started or isinstance(self._returncode, int):
            return self._returncode

        assert self._proc is not None
        assert self._start_time is not None

        self._returncode = self._proc.poll()
        if self._returncode is not None:
            self._end_time = time.time()

        elif (
            self._timeout is not None and
            time.time() - self._start_time > self._timeout
        ):
            assert self._cmd is not None
            raise subprocess.TimeoutExpired(cmd=self._cmd, timeout=self._timeout)

        return self._returncode

    @property
    def language(self) -> Language:
        return self._timer_args.language

    @property
    def result(self) -> Optional[WorkerOutput]:
        self._collect()
        return self._output

    @property
    def worker_failure(self) -> Optional[WorkerFailure]:
        self._collect()
        return self._worker_failure

    def _collect(self) -> None:
        if not self.ready:
            return

        # Cannot both succeed and fail.
        assert self._output is None or self._worker_failure is None

        if self._output is not None or self._worker_failure is not None:
            # Already collected.
            return

        returncode = self.returncode
        assert isinstance(returncode, int)
        assert self._proc is not None
        assert self._communication_file is not None

        result: Union[WorkerOutput, WorkerFailure]
        with open(self._communication_file, "rb") as f:
            result = pickle.load(f)
            assert isinstance(result, (TimerArgs, WorkerOutput, WorkerFailure))

        if isinstance(result, WorkerOutput):
            if returncode:
                # Worker managed to complete the designated task, but worker
                # process did not finish cleanly.
                self._worker_failure = WorkerFailure(
                    "Worker failed, but did not return diagnostic information.")
            else:
                self._output = result

        elif isinstance(result, TimerArgs):
            # Worker failed, but did not write a result so we're left with the
            # original TimerArgs. Grabbing all of stdout and stderr isn't
            # ideal, but we don't have a better way to determine what to keep.
            proc_stdout = self._proc.stdout
            assert proc_stdout is not None
            self._worker_failure = WorkerFailure(
                failure_trace=proc_stdout.read().decode("utf-8"))

        else:
            assert isinstance(result, WorkerFailure)
            self._worker_failure = result

        # Release communication file.
        os.remove(self._communication_file)
        self._communication_file = None

    @property
    def run_time(self) -> float:
        assert self._end_time is not None
        return self._end_time - self.start_time

    @property
    def start_time(self) -> float:
        assert self._start_time is not None
        return self._start_time

    def start(self, cpu_list: Optional[str] = None) -> None:
        assert not self.started, "Already started."
        self._communication_file = os.path.join(
            get_temp_dir(), f"{uuid.uuid4()}.pkl")

        with open(self._communication_file, "wb") as f:
            assert isinstance(self._timer_args, TimerArgs)
            pickle.dump(self._timer_args, f)

        cmd: List[str] = []
        if self._source_cmd is not None:
            cmd.extend([self._source_cmd, "&&"])

        if cpu_list is not None:
            cmd.extend(["taskset", "--cpu-list", cpu_list])

        cmd.extend([
            "python", "-m", "execution.worker",
            "--communication_file", self._communication_file,
        ])

        self._cmd = " ".join(cmd)
        self._proc = subprocess.Popen(
            self._cmd,
            cwd=os.path.split(os.path.split(os.path.abspath(__file__))[0])[0],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
        )
        self._start_time = time.time()

    def interrupt(self) -> None:
        """Soft interrupt. Allows subprocess to cleanup."""
        if self._proc is not None:
            self._proc.send_signal(signal.SIGINT)

    def terminate(self) -> None:
        """Hard interrupt. Immediately SIGKILL subprocess."""
        if self._proc is not None:
            self._proc.terminate()
