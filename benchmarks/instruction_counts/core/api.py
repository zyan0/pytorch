"""Key enums and structs used to handle data flow within the benchmark."""
import dataclasses
import enum
import re
import textwrap
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    # Benchmark utils are only partially strict compliant, so MyPy won't follow
    # imports using the public namespace. (Due to an exclusion rule in
    # mypy-strict.ini)
    from torch.utils.benchmark.utils.common import Measurement
    from torch.utils.benchmark.utils.timer import Language
    from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import CallgrindStats
else:
    from torch.utils.benchmark import CallgrindStats, Language, Measurement


# =============================================================================
# == Benchmark definition =====================================================
# =============================================================================
class Setup(enum.Enum):
    """Defines the class of setup that a stmt requires.

    Because a GroupedTimerArgs may (and generally will) represent both Python
    and C++ code, we chunk setup into broad groups which are resolved into
    language specific strings. This also results in more compact and readable
    definitions.
    """
    NONE = 0
    GENERIC = 1
    MESOSCALE = 2
    AUTOGRAD = 3
    EXAMPLE_FOR_ADHOC = 4


class Mode(enum.Enum):
    # Generated from GroupedTimerArgs
    PY = "Python"
    CPP = "C++"
    PY_TS = "Python (TorchScript)"
    CPP_TS = "C++ (TorchScript)"

    # TimerArgs was explicitly provided.
    EXPLICIT = "Explicit"


class CostEstimate(enum.Enum):
    """Hint for how expensive a benchmark is expected to be.

    Timer supports adaptive timing for wall times, but not instruction counts.
    Generally this is desired since we want deterministic instruction counts,
    however it can be tedious to choose sensible numbers when defining a slew
    of benchmarks.
    """
    AUTO = 0
    LESS_THAN_10_US = 1
    LESS_THAN_50_US = 2
    LESS_THAN_100_US = 3
    LESS_THAN_250_US = 4
    LESS_THAN_1000_US = 5
    GIANT = 6


@dataclasses.dataclass(frozen=True)
class TimerArgs:
    """Container for Timer constructor arguments.

    This dataclass serves two roles. First, it is a simple interface for
    defining benchmarks. (See GroupedTimerArgs for the advanced interface.)
    Second, it provides serialization for controlling workers. `Timer` is not
    pickleable, so instead the parent process will pass `TimerArgs` instances
    to workers for processing.
    """

    # Timer constructor arguments.
    stmt: str
    setup: str
    num_threads: Union[int, Tuple[int, ...]] = 1
    language: Language = Language.PYTHON

    # Unlike `adaptive_autorange`, `collect_callgrind` does not dynamically
    # adjust based on the cost of a stmt, so we must either provide a cost
    # estimate or tell workers to determine a sensible value.
    cost: CostEstimate = CostEstimate.AUTO

    def flatten(self) -> Tuple["TimerArgs", ...]:
        if isinstance(self.num_threads, int):
            return (self,)

        return tuple(
            dataclasses.replace(self, num_threads=num_threads)
            for num_threads in self.num_threads)


@dataclasses.dataclass(frozen=True)
class GroupedTimerArgs:
    """Defines a set of related benchmarks which are semantically equivalent.

    There are four ways one might reasonably wish to run a PyTorch snippet:
      - Using the Python eager API
      - Using the C++ eager frontend
      - Running a TorchScript model eagerly from Python
      - Running a TorchScript model which has been loaded into C++

    It is useful to define them together, both for clairity when reading
    benchmark definitions and for later processing and analysis.

    We may, of course, only be interested in a subset of cases. For instance we
    may be benchmarking Python code which does not have a C++ analog, or a
    statement which is not TorchScript-able. This is supported by simply
    omitting arguments.

    In order to measure TorchScript performance, `py_stmt` must be specified
    and must be scriptable. (It will be scripted, not traced.) Secondly,
    `signature` must be specified and take the form `f(args) -> output`. e.g.

        "f(a, b, c) -> d"
        "f(x) -> None"

    This is used to build both the model and invocation. Note that the return
    is a literal variable, not a type. TorchScript will optimize away
    computation which does not have observable side effects, so some functions
    need to return a result to actually benchmark the task of interest.

    Example:
    ```
    GroupedTimerArgs(
        setup=Setup.GENERIC,  # Creates a float Tensor `x`
        py_stmt="y = x + x.t()",
        cpp_stmt="auto y = x + x.t();",

        # Optional. If present, we can make a TorchScript function as well.
        signature="f(x) -> y",
    )
    ```

    GroupedTimerArgs will ultimately be parsed down to one or more TimerArgs
    for evaluation.
    """
    setup: Setup
    py_stmt: Optional[str] = None
    cpp_stmt: Optional[str] = None
    signature: Optional[str] = None
    num_threads: Union[int, Tuple[int, ...]] = 1
    cost: CostEstimate = CostEstimate.AUTO

    def __post_init__(self) -> None:
        # This is done purely to improve readability.
        if self.py_stmt is not None:
            object.__setattr__(self, "py_stmt", textwrap.dedent(self.py_stmt).strip())

        if self.cpp_stmt is not None:
            object.__setattr__(self, "cpp_stmt", textwrap.dedent(self.cpp_stmt).strip())

        if self.py_stmt is None and self.cpp_stmt is None:
            raise ValueError("You must specify at least one of `py_stmt`, `cpp_stmt`")

        # Check that signature is valid.
        self.torchscript_signature

    @property
    def torchscript_signature(self) -> Optional[Tuple[Tuple[str, ...], str]]:
        if self.signature is None:
            return None

        if self.py_stmt is None:
            # `py_stmt` populates the body of the function.
            raise ValueError("signature provided, but `py_stmt` is None.")

        match = re.search(r"^f\((.*)\) -> (.*)$", self.signature)
        if match is None:
            raise ValueError(f"Invalid signature: `{self.signature}`")

        return tuple(match.groups()[0].split(", ")), match.groups()[1].strip()


# =============================================================================
# == Benchmark evaluation =====================================================
# =============================================================================
@dataclasses.dataclass(frozen=True)
class WorkerOutput:
    wall_time: Measurement
    instructions: CallgrindStats
    cost: CostEstimate  # Emperical cost.


@dataclasses.dataclass(frozen=True)
class WorkerFailure:
    # If a worker fails, we attach the string contents of the Exception
    # rather than the Exception object itself. This is done for two reasons:
    #   1) Depending on the type thrown, `e` may or may not be pickleable
    #   2) If we re-throw in the main process, we lose the true stack trace.
    failure_trace: str


class WorkerFailed(Exception):
    """Raised in the main process when a worker failure is detected."""
    def __init__(
        self,
        timer_args: TimerArgs,
        wrapped_trace: Optional[str] = None
    ) -> None:
        self.timer_args: TimerArgs = timer_args
        self.wrapped_trace: Optional[str] = wrapped_trace
        super().__init__()
