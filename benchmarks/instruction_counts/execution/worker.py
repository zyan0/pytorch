"""File invoked through subprocess to actually carry out measurements.

`worker.py` is deliberately isolated from the rest of the benchmark
infrastructure. Its only import is `core.api`, which in turn simply defines
some communication structs and enums. The life of a worker is very simple:
it receives a file containing a `WorkerInput` telling it what to run,
and writes a `WorkerOutput` result back to the same file.

Because this file only expects to run in a child context, error handling means
plumbing failures up to the caller, not raising in this process.
"""
import argparse
import io
import pickle
from typing import Dict, Tuple, Union, TYPE_CHECKING
import traceback
import sys

from core.api import CostEstimate, TimerArgs, WorkerFailure, WorkerOutput

if TYPE_CHECKING:
    # Benchmark utils are only partially strict compliant, so MyPy won't follow
    # imports using the public namespace. (Due to an exclusion rule in
    # mypy-strict.ini)
    from torch.utils.benchmark.utils.common import Measurement
    from torch.utils.benchmark.utils.timer import Timer
    from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import CallgrindStats
else:
    from torch.utils.benchmark import CallgrindStats, Measurement, Timer


# While the point of this is mainly to collect instruction counts, we're going
# to have to compile C++ timers anyway (as they're used as a check before
# calling Valgrind), so we may as well grab wall times for reference. They
# are comparatively inexpensive.
MIN_RUN_TIME = 5


# Heuristics for estimating the runtime of `collect_callgrind` based on
# measured wall time:
#   t_callgrind ~= 40 + 50 * t_wall
# The overhead (40 seconds) is startup and post processing time, while the
# (rather onerous) factor of 50 reflects the slowness of the Valgrind
# virtual machine and Callgrind instrumentation. (As well as instruction post
# processing.)

CALLGRIND_COST_GUIDE: Dict[CostEstimate, Tuple[float, int]] = {
    # Key: (max wall time, callgrind number)
    CostEstimate.LESS_THAN_10_US: (10e-6, 25_000),
    CostEstimate.LESS_THAN_50_US: (50e-6, 5_000),
    CostEstimate.LESS_THAN_100_US: (100e-6, 2_500),
    CostEstimate.LESS_THAN_250_US: (250e-6, 1_000),
    CostEstimate.LESS_THAN_1000_US: (1e-3, 250),
    CostEstimate.GIANT: (1e9, 10),  # Catch all
}

# Ensure map is complete.
assert tuple(CostEstimate) == (CostEstimate.AUTO,) + tuple(CALLGRIND_COST_GUIDE.keys())

# Ensure map is strictly increasing.
assert all(c1 > c0 for (c0, _), (c1, _) in
    zip(CALLGRIND_COST_GUIDE.values(), list(CALLGRIND_COST_GUIDE.values())[1:]))


def make_timer(args: TimerArgs) -> Timer:
    """Timer class is not serializable."""
    assert isinstance(args.num_threads, int)
    return Timer(
        stmt=args.stmt,
        setup=args.setup,
        num_threads=args.num_threads,
        language=args.language,
    )


def _run(timer_args: TimerArgs) -> WorkerOutput:
    timer = make_timer(timer_args)
    m = timer.blocked_autorange(min_run_time=MIN_RUN_TIME)

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
        instructions=stats,
        cost=cost,
    )


def main(communication_file: str) -> None:
    result: Union[WorkerOutput, WorkerFailure]
    try:
        with open(communication_file, "rb") as f:
            timer_args: TimerArgs = pickle.load(f)
            assert isinstance(timer_args, TimerArgs)
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
