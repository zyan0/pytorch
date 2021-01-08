"""This module sets the user facing command APIs, both CLI and programmatic."""
import dataclasses
import itertools as it
import textwrap
from typing import cast, Dict, Iterable, Mapping, List, Optional, Tuple, TYPE_CHECKING

from core.api import Mode
from core.types import Label
from core.unpack_groups import unpack
from definitions.ad_hoc import ADHOC_BENCHMARKS
from definitions.standard import BENCHMARKS
from execution.callback import FixedReplicateCallback
from execution.future import WorkOrder
from execution.runner import Runner
from frontend.display import render_ab, ResultType, ValueType
from worker.main import CostEstimate, WorkerTimerArgs, WorkerOutput

if TYPE_CHECKING:
    # See core.api for an explanation why this is necessary.
    from torch.utils.benchmark.utils.common import Measurement
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language, Measurement


def _make_sentry(source_cmd: Optional[str]) -> WorkOrder:
    """Known stable tasks which are injected to test measurement stability."""
    timer_args = WorkerTimerArgs(
        stmt="""
            auto x = torch::ones({4, 4});
            auto z = x - y;
        """,
        setup="auto y = torch::ones({4, 4});",
        num_threads=1,
        language=Language.CPP,
        cost=CostEstimate.LESS_THAN_10_US,
    )

    return WorkOrder(
        label=("Impl", "Sentry"),
        mode=Mode.EXPLICIT_CPP,
        timer_args=timer_args,
        source_cmd=source_cmd,
        timeout=180.0,
        retries=2,
    )


def _collect(
    source_cmds: Tuple[Optional[str], ...] = (None,),
    timing_replicates: int = 0,
    ad_hoc: bool = False
) -> Tuple[ResultType, ...]:
    all_work_items: List[WorkOrder] = []
    work_items_by_source_cmd: List[Tuple[WorkOrder, ...]] = []

    # Set up normal benchmarks
    benchmarks = ADHOC_BENCHMARKS if ad_hoc else BENCHMARKS
    for label, mode, timer_args in unpack(benchmarks):
        orders: Tuple[WorkOrder, ...] = tuple(
            WorkOrder(
                label=label,
                mode=mode,
                timer_args=timer_args,
                source_cmd=source_cmd,
                timeout=180.0,
                retries=2,
            )
            for source_cmd in source_cmds
        )
        work_items_by_source_cmd.append(orders)
        all_work_items.extend(orders)

    # Set up sentry measurements for warnings.
    sentry_work_items: List[Tuple[WorkOrder, ...]] = [
        tuple(_make_sentry(source_cmd) for _ in range(3))
        for source_cmd in source_cmds
    ]
    all_work_items = list(it.chain(*sentry_work_items)) + all_work_items

    # Collect measurements.
    time_callback = FixedReplicateCallback(
        num_replicates=timing_replicates,
        work_items_by_source_cmd=tuple(work_items_by_source_cmd),
    )
    runner = Runner(work_items=tuple(all_work_items), callbacks=(time_callback,))
    results = runner.run()

    # Warn if there is ANY variation in instruction counts. While Python has
    # some jitter, C++ should be truly detministic.
    for source_cmd, work_items in zip(source_cmds, sentry_work_items):
        sentry_results = [results[w].instructions.counts() for w in work_items]
        if len(set(sentry_results)) > 1:
            print(textwrap.dedent(f"""
                WARNING: measurements are unstable. (source cmd: `{source_cmd}`)
                    Three C++ sentries were run which should have been completely
                    deterministic, but instead resulted in the following counts:
                      {sentry_results}
            """))

    # Organize normal benchmark results.
    output: List[ResultType] = []
    for work_items in zip(*work_items_by_source_cmd):
        output_i: List[Tuple[Label, int, Mode, ValueType]] = []
        for w in work_items:
            r = results[w]
            output_i.append((
                w.label,
                w.timer_args.num_threads,
                w.mode,
                (r.instructions, time_callback.get_times_for(w))
            ))
        output.append(tuple(output_i))

    return tuple(output)


def ab_test(source_a: str, source_b: str, timing_replicates: int = 0, ad_hoc: bool = False) -> None:
    results = _collect(
        source_cmds=(source_a, source_b),
        timing_replicates=timing_replicates,
        ad_hoc=ad_hoc,
    )

    def render() -> None:
        import importlib
        import frontend.display
        importlib.reload(frontend.display)
        frontend.display.render_ab(results[0], results[1])


    import pdb
    pdb.set_trace()
