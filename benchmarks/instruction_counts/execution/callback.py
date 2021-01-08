"""Dynamically control Runner jobs."""
import dataclasses
from typing import Dict, Iterable, List, Set, Tuple, TYPE_CHECKING

from execution.future import WorkOrder
from worker.main import WorkerOutput

if TYPE_CHECKING:
    # See core.api for an explanation.
    from torch.utils.benchmark.utils.common import Measurement
else:
    from torch.utils.benchmark import Measurement


class Callback:
    def __call__(
        self,
        work_order: WorkOrder,
        output: WorkerOutput
    ) -> Iterable[WorkOrder]:
        return ()


class FixedReplicateCallback(Callback):
    def __init__(
        self,
        num_replicates: int,
        work_items_by_source_cmd: Tuple[Tuple[WorkOrder, ...], ...],
    ) -> None:
        assert num_replicates >= 0
        self._num_replicates = num_replicates
        self._work_items_by_source_cmd = work_items_by_source_cmd

        self._results: Dict[WorkOrder, WorkerOutput] = {}
        self._primary_indices: Dict[WorkOrder, Tuple[int, int]] = {
            wij: (i, j)
            for i, wi in enumerate(work_items_by_source_cmd)
            for j, wij in enumerate(wi)
        }

        # Map "timing only" runs to the original work item.
        self._secondary_indices: Dict[WorkOrder, Tuple[int, int]] = {}
        self._inverse_secondary_indices: Dict[Tuple[int, int], List[WorkOrder]] = {
            ij: [] for ij in self._primary_indices.values()
        }

        n_source_cmds = len(work_items_by_source_cmd[0])
        assert all(len(w) == n_source_cmds for w in work_items_by_source_cmd)
        self._outstanding = [n_source_cmds for _ in work_items_by_source_cmd]

    def __call__(
        self,
        work_order: WorkOrder,
        output: WorkerOutput,
    ) -> Iterable[WorkOrder]:
        assert work_order not in self._results
        self._results[work_order] = output

        if work_order not in self._primary_indices:
            return ()

        i, j = self._primary_indices[work_order]
        self._outstanding[i] -= 1
        assert self._outstanding[i] >= 0

        if self._outstanding[i]:
            return ()

        work_items = self._work_items_by_source_cmd[i]
        counts: Tuple[int, ...] = tuple(
            self._results[w].instructions.counts(denoise=True)
            for w in work_items
        )

        new_work_items: List[WorkOrder] = []
        if len(set(counts)) > 1:
            for j, w in list(enumerate(work_items)) * self._num_replicates:
                new_timer_args = dataclasses.replace(
                    w.timer_args,
                    collect_instructions=False,
                )

                timing_w = dataclasses.replace(w, timer_args=new_timer_args)
                self._secondary_indices[timing_w] = (i, j)
                self._inverse_secondary_indices[(i, j)].append(timing_w)
                new_work_items.append(timing_w)

        return new_work_items

    def get_times_for(self, w: WorkOrder) -> Measurement:
        ij = self._primary_indices[w]
        t = Measurement.merge(
            [self._results[w].wall_time] +
            [self._results[wi].wall_time
             for wi in self._inverse_secondary_indices[ij]]
        )
        assert len(t) == 1
        return t[0]










# class TimeReplicateCallback(Callback):
#     def __init__(
#         self,
#         work_items_by_source_cmd: Tuple[Tuple[WorkOrder, ...], ...],
#         active: bool = True,
#     ) -> None:
#         self._active: bool = active

#         n_source_cmds = len(work_items_by_source_cmd[0])
#         assert all(len(w) == n_source_cmds for w in work_items_by_source_cmd)

#         self._outstanding = [n_source_cmds for _ in work_items_by_source_cmd]
#         self._results: Dict[WorkOrder, WorkerOutput] = {}
#         self._work_items_by_source_cmd = work_items_by_source_cmd
#         self._indices: Dict[WorkOrder, Tuple[int, int]] = {
#             wij: (i, j)
#             for i, wi in enumerate(work_items_by_source_cmd)
#             for j, wij in enumerate(wi)
#         }

#         # Map "timing only" runs to the original work item.
#         self._secondary_indices: Dict[WorkOrder, Tuple[int, int]] = {}

#         self._times: Dict[Tuple[int, int], Measurement] = {}
#         self._converged: Set[Tuple[int, int], bool] = set()
#         self._num_measurements: Dict[Tuple[int, int], int] = {
#             k: 0 for k in self._indices.values()}

#     def __call__(
#         self,
#         work_order: WorkOrder,
#         output: WorkerOutput,
#     ) -> Iterable[WorkOrder]:
#         assert work_order not in self._results
#         self._results[work_order] = output

#         if work_order in self._indices:
#             # Primary task.
#             i, j = self._indices[work_order]

#         elif work_order in self._secondary_indices:
#             # Timing replicate.
#             i, j = self._secondary_indices[work_order]

#         else:
#             # Sentry task
#             return ()

#         self._num_measurements[(i, j)] += 1
#         if (i, j) in self._times:
#             assert not self._outstanding[i], f"{self._outstanding}"
#             t = Measurement.merge([output.wall_time, self._times[(i, j)]])
#             assert len(t) == 1
#             self._times[(i, j)] = t[0]
#         else:
#             assert self._outstanding[i]
#             assert self._num_measurements[(i, j)] == 1
#             self._times[(i, j)] = output.wall_time

#         if not self._active:
#             # Don't schedule new tasks if Callback is not "active"
#             return ()

#         self._outstanding[i] = max(self._outstanding[i] - 1, 0)
#         if self._outstanding[i]:
#             return ()

#         work_items = self._work_items_by_source_cmd[i]
#         new_work_items: List[WorkOrder] = []
#         if all(self._num_measurements[self._indices[w]] == 1 for w in work_items):
#             # First time all `n_source_cmds` have finished.
#             counts: Tuple[int, ...] = tuple(
#                 self._results[w].instructions.counts(denoise=True)
#                 for w in work_items)

#             # Start with two replicates.
#             for j, w in list(enumerate(work_items)) * 2:
#                 new_timer_args = dataclasses.replace(
#                     w.timer_args,
#                     collect_instructions=False,
#                 )

#                 timing_w = dataclasses.replace(w, timer_args=new_timer_args)
#                 self._secondary_indices[timing_w] = (i, j)
#                 new_work_items.append(timing_w)

#         elif self._times[(i, j)].significant_figures >= 3:
#             # Success.
#             self._converged.add((i, j))

#         elif self._num_measurements[(i, j)] >= 30:
#             # Give up.
#             pass

#         else:
#             new_timer_args = dataclasses.replace(
#                 work_order.timer_args,
#                 collect_instructions=False,
#             )

#             timing_w = dataclasses.replace(work_order, timer_args=new_timer_args)
#             self._secondary_indices[timing_w] = (i, j)
#             new_work_items.append(timing_w)

#         return new_work_items

#     def get_times_for(self, w: WorkOrder) -> Measurement:
#         return self._times[self._indices[w]]
