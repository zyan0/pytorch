import dataclasses
import enum
import multiprocessing
import os
from typing import Optional, Tuple, Union

from torch.utils.benchmark import Measurement, CallgrindStats


from tasks.spec import TimerSpec

RUNNER_ROOT = os.path.split(os.path.abspath(__file__))[0]
WORKER_PATH = os.path.join(RUNNER_ROOT, 'worker.py')
NUM_CORES = multiprocessing.cpu_count()
NUM_WORKERS = 8  # max(NUM_CORES - 4, 1)
CALLGRIND_NUMBER =  100_000
CALLGRIND_TIMEOUT = 120
MIN_RUN_TIME = 20


class MeasurementType(enum.Enum):
    CALLGRIND = 0
    WALL_TIME = 1


@dataclasses.dataclass(frozen=True)
class WorkerInput:
    measurement_type: MeasurementType
    tasks: Tuple[TimerSpec, ...]


@dataclasses.dataclass(frozen=True)
class WorkerOutput:
    results: Union[
        Tuple[Measurement, ...],
        Tuple[CallgrindStats, ...]
    ]

    # If a worker fails, we attach the string contents of the Exception
    # rather than the Exception object itself. This is done for two reasons:
    #   1) Depending on the type thrown, e may or may not be pickleable
    #   2) If we re-throw in the main process, we lose traceback info.
    e_str: Optional[str] = None
