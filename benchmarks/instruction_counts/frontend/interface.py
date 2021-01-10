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
    runner = Runner(work_items=tuple(all_work_items))
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
                (r.instructions, r.wall_time)
            ))
        output.append(tuple(output_i))

    return tuple(output)


def demo() -> None:
    envs = (
        "ab_ref",       # fcb69d2ebaede960e7708706436d372b68807921
        "ab_change_0",  # eef5eb05bf0468ed5f840d2bf3e09c135b8760df
        "ab_change_1",  # dde5b6e177ec24d34651ffd8df04b4ebdf264e6e
    )

    source_cmds = tuple(f"source activate {env}" for env in envs)


    import os
    import shutil
    import subprocess

    ref_path = os.path.abspath(__file__)
    for _ in range(4):
        ref_path = os.path.split(ref_path)[0]
    ref_path = os.path.join(ref_path, "torch/utils/benchmark/utils")

    for env, source_cmd in zip(envs, source_cmds):
        torch_path = subprocess.run(
            f"{source_cmd} && python -c 'import torch;print(torch.__file__)'",
            stdout=subprocess.PIPE,
            shell=True,
            encoding="utf-8",
        ).stdout
        benchmark_path = os.path.join(os.path.split(torch_path)[0], "utils/benchmark/utils")

        print(f"Patching Timer: `{env}``")
        if os.path.exists(benchmark_path):
            shutil.rmtree(benchmark_path)
        shutil.copytree(ref_path, benchmark_path)

    results = _collect(
        source_cmds=source_cmds,
        timing_replicates=0,
        ad_hoc=False,
    )

    def render(ia: int = 0, ib: int = 1) -> None:
        import importlib
        import frontend.display
        importlib.reload(frontend.display)
        print("\n" * 10)
        frontend.display.render_ab(results[ia], results[ib])

    import pdb
    pdb.set_trace()



def ab_test(source_a: str, source_b: str, timing_replicates: int = 0, ad_hoc: bool = False) -> None:
    results = _collect(
        source_cmds=(
            source_a,
            source_b,
            source_a,  # For debug A/A testing
        ),
        timing_replicates=timing_replicates,
        ad_hoc=ad_hoc,
    )

    def render(ia: int = 0, ib: int = 1, display_time: bool = False) -> None:
        import importlib
        import frontend.display
        importlib.reload(frontend.display)
        print("\n" * 10)
        frontend.display.render_ab(results[ia], results[ib], display_time)


    import pdb
    pdb.set_trace()
