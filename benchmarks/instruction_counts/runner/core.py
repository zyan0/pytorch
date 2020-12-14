import dataclasses
import enum
import multiprocessing
import os
from typing import Optional, Tuple, Union

from torch.utils.benchmark import Measurement, CallgrindStats


from tasks.spec import TimerSpec


WORKER_PATH = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'worker.py')
NUM_CORES = multiprocessing.cpu_count()
NUM_WORKERS = max(NUM_CORES - 4, 1)
CALLGRIND_NUMBER =  100_000
CALLGRIND_TIMEOUT = 120
MIN_RUN_TIME = 5


class MeasurementType(enum.Enum):
    CALLGRIND = 0
    WALL_TIME = 1


@dataclasses.dataclass(frozen=True)
class WorkerInput:
    measurement_type: MeasurementType
    tasks = Tuple[TimerSpec, ...]


@dataclasses.dataclass(frozen=True)
class WorkerOutput:
    results: Union[
        Tuple[Measurement, ...],
        Tuple[CallgrindStats, ...]
    ]
    e: Optional[Exception] = None
