"""File invoked through subprocess to actually carry out measurements.

`worker/main.py` is deliberately isolated from the rest of the benchmark
infrastructure. Other parts of the benchmark rely on this file, but
`worker/` has only one Python file does not import ANYTHING from the rest
of the benchmark suite. The reason that this is important is that we can't
rely on paths to access the other files (namely `core.api`) since a source
command might change the CWD. It also helps keep startup time down by limiting
spurious definition work.

The life of a worker is very simple:
    It receives a file containing a `WorkerTimerArgs` telling it what to run,
    and writes a `WorkerOutput` result back to the same file.

Because this file only expects to run in a child context, error handling means
plumbing failures up to the caller, not raising in this process.
"""
import argparse
import dataclasses
import enum
import io
import os
import pickle
from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING
import traceback
import sys

if TYPE_CHECKING:
    # Benchmark utils are only partially strict compliant, so MyPy won't follow
    # imports using the public namespace. (Due to an exclusion rule in
    # mypy-strict.ini)
    from torch.utils.benchmark.utils.common import Measurement
    from torch.utils.benchmark.utils.timer import Language, Timer
    from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import CallgrindStats
else:
    from torch.utils.benchmark import CallgrindStats, Language, Measurement, Timer

WORKER_PATH = os.path.abspath(__file__)


# =============================================================================
# == Interfaces ===============================================================
# =============================================================================
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
class WorkerTimerArgs:
    """Mirrors core.api.TimerArgs

    Note that `num_threads` is narrowed from `Union[int, Tuple[int, ...]]` to
    `int`. `core.api` will assert that `WorkerTimerArgs` matches `TimerArgs`.
    """
    stmt: str
    setup: str
    num_threads: int
    language: Language
    cost: CostEstimate
    collect_instructions: bool = True

    @classmethod
    def keys(cls) -> Tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(cls))


@dataclasses.dataclass(frozen=True)
class WorkerOutput:
    wall_time: Measurement
    _instructions: Optional[CallgrindStats]
    cost: CostEstimate  # Emperical cost. (If AUTO.)

    @property
    def instructions(self) -> CallgrindStats:
        # We sometimes omit instruction collection; in such cases the caller
        # should know not to ask for them.
        assert self._instructions is not None
        return self._instructions


@dataclasses.dataclass(frozen=True)
class WorkerFailure:
    # If a worker fails, we attach the string contents of the Exception
    # rather than the Exception object itself. This is done for two reasons:
    #   1) Depending on the type thrown, `e` may or may not be pickleable
    #   2) If we re-throw in the main process, we lose the true stack trace.
    failure_trace: str


class WorkerUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        """Resolve import for pickle.

        When the main runner uses a symbol `foo` from this file, it sees it as
        `worker.main.foo`. However the worker (called as a standalone file)
        sees the same symbol as `__main__.foo`. We have to help pickle
        understand where to look for the interface symbols.
        """
        return {
            "CostEstimate": CostEstimate,
            "WorkerTimerArgs": WorkerTimerArgs,
            "WorkerOutput": WorkerOutput,
            "WorkerFailure": WorkerFailure,
        }.get(name, None) or super().find_class(module, name)

    def load_from_worker(self) -> Union[WorkerTimerArgs, WorkerOutput, WorkerFailure]:
        result = self.load()
        assert isinstance(result, (WorkerTimerArgs, WorkerOutput, WorkerFailure))
        return result


# Heuristics for estimating the runtime of `collect_callgrind` based on
# measured wall time:
#   t_callgrind ~= 40 + 50 * t_wall
# The overhead (40 seconds) is startup and post processing time, while the
# (rather onerous) factor of 50 reflects the slowness of the Valgrind
# virtual machine and Callgrind instrumentation. (As well as instruction post
# processing.)
CALLGRIND_COST_GUIDE: Dict[CostEstimate, Tuple[float, int]] = {
    # Key: (max wall time, callgrind number)
    CostEstimate.LESS_THAN_10_US: (10e-6, 50_000),
    CostEstimate.LESS_THAN_50_US: (50e-6, 10_000),
    CostEstimate.LESS_THAN_100_US: (100e-6, 5_000),
    CostEstimate.LESS_THAN_250_US: (250e-6, 2_000),
    CostEstimate.LESS_THAN_1000_US: (1e-3, 500),
    CostEstimate.GIANT: (1e9, 10),  # Catch all
}

# Ensure map is complete.
assert tuple(CostEstimate) == (CostEstimate.AUTO,) + tuple(CALLGRIND_COST_GUIDE.keys())

# Ensure map is strictly increasing.
assert all(c1 > c0 for (c0, _), (c1, _) in
    zip(CALLGRIND_COST_GUIDE.values(), list(CALLGRIND_COST_GUIDE.values())[1:]))


MIN_RUN_TIME = 5


# =============================================================================
# == Execution ================================================================
# =============================================================================
def _run(timer_args: WorkerTimerArgs) -> WorkerOutput:
    timer = Timer(
        stmt=timer_args.stmt,
        setup=timer_args.setup,
        num_threads=timer_args.num_threads,
        language=timer_args.language,
    )

    if timer_args.collect_instructions:
        # While the point of this is mainly to collect instruction counts,
        # we're going to have to compile C++ timers anyway (as they're used as
        # a check before calling Valgrind), so we may as well grab wall times
        # for reference. They are comparatively inexpensive.
        m = timer.blocked_autorange(min_run_time=MIN_RUN_TIME)
    else:
        # Timing only run, so we can run for much longer.
        m = timer.blocked_autorange(min_run_time=60)
        return WorkerOutput(wall_time=m, _instructions=None, cost=timer_args.cost)

    cost: CostEstimate = timer_args.cost
    n: int
    if cost == CostEstimate.AUTO:
        t: float = m.median
        for cost, (t_max, n) in CALLGRIND_COST_GUIDE.items():
            if t <= t_max:
                break
    else:
        n = CALLGRIND_COST_GUIDE[cost][1]

    stats = timer.collect_callgrind(number=n, collect_baseline=False)
    return WorkerOutput(
        wall_time=m,
        _instructions=stats,
        cost=cost
    )


def main(communication_file: str) -> None:
    result: Union[WorkerOutput, WorkerFailure]
    try:
        with open(communication_file, "rb") as f:
            timer_args: WorkerTimerArgs = WorkerUnpickler(f).load()
            assert isinstance(timer_args, WorkerTimerArgs)
        result = _run(timer_args)

    except KeyboardInterrupt:
        # Runner process sent SIGINT.
        sys.exit()

    except:
        trace_f = io.StringIO()
        traceback.print_exc(file=trace_f)
        result = WorkerFailure(failure_trace=trace_f.getvalue())

    with open(communication_file, "wb") as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--communication_file', type=str)
    communication_file = parser.parse_args().communication_file
    main(communication_file)
