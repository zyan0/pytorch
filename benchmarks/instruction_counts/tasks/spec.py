
import dataclasses
import enum
import textwrap
from typing import Optional, Tuple

from torch.utils.benchmark import Language, Timer


__all__ = ["Setup", "TaskSpec", "TimerSpec"]


class Setup(enum.Enum):
    NONE = 0
    GENERIC = 1
    MESOSCALE = 2
    AUTOGRAD = 3


class Mode(enum.Enum):
    PY = "Python"
    CPP = "C++"
    PY_TS = "Python (TorchScript)"
    CPP_TS = "C++ (TorchScript)"


@dataclasses.dataclass()
class TaskSpec:
    setup: Setup
    py_stmt: str
    cpp_stmt: str
    torchscript_args: Optional[Tuple[str, ...]] = None
    torchscript_return: Optional[str] = None

    def __post_init__(self):
        self.py_stmt = textwrap.dedent(self.py_stmt).strip()
        self.cpp_stmt = textwrap.dedent(self.cpp_stmt).strip()


@dataclasses.dataclass(frozen=True)
class TimerSpec:
    """Timers are not pickleable, so we define a dataclass to send to workers."""
    stmt: str
    setup: str
    language: Language


def make_timer(timer_spec: TimerSpec):
    return Timer(
        stmt=timer_spec.stmt,
        setup=timer_spec.setup,
        language=timer_spec.language,
    )
