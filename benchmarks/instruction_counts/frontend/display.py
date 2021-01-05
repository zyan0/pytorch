import abc
import dataclasses
import enum
import itertools as it
import statistics
from typing import cast, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from core.api import Mode
from core.types import Label
from worker.main import WorkerOutput

if TYPE_CHECKING:
    # See core.api for an explanation why this is necessary.
    from torch.utils.benchmark.utils.common import Measurement
    from torch.utils.benchmark.utils.timer import Language
    from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import CallgrindStats
else:
    from torch.utils.benchmark import CallgrindStats, Language, Measurement


ValueType = Tuple[CallgrindStats, Tuple[Measurement, ...]]
ResultType = Tuple[Tuple[Label, int, Mode, ValueType], ...]


class Alignment(enum.Enum):
    LEFT = 0
    RIGHT = 1
    CENTER = 2
    ANY = 3


class ShouldRender(enum.Enum):
    YES = 0
    MAYBE = 1


class Cell(abc.ABC):
    # TODO: annotate with `Final` and `final`.
    _row: Optional["Row"] = None

    def set_context(self, row: "Row") -> None:
        self._row = row

    def col_reduce(self, f: Callable[[Tuple["Cell", ...]], Any]) -> Tuple[Any, int]:
        assert self._row is not None
        return self._row._table.col_reduce(self, f)

    @abc.abstractmethod
    def render(self) -> str:
        ...

    @abc.abstractmethod
    def alignment(self) -> Alignment:
        ...

    @property
    def should_render(self) -> ShouldRender:
        return ShouldRender.YES


@dataclasses.dataclass(frozen=True)
class Row:
    cells: Tuple[Cell, ...]
    _table: "Table"

    def __len__(self) -> int:
        return len(self.cells)

    def render(self) -> Tuple[Tuple[str, ...], ...]:
        col_lines: List[Tuple[str, ...]] = [
            tuple(c.render().splitlines(keepends=False)) for c in self.cells]
        row_h = max(len(lines) for lines in col_lines)
        return tuple(("",) * (row_h - len(lines)) + lines for lines in col_lines)


class Table:
    _rows: Tuple[Row, ...]
    _reduction_cache: Dict[Tuple[int, int], Any]
    _row_index_map: Dict[int, int]
    _col_index_map: Dict[int, int]

    def __init__(self, rows: Iterable[Tuple[Cell, ...]]):
        self._reduction_cache = {}
        self._rows = tuple(Row(r, self) for r in rows)
        self._row_index_map = {}
        self._col_index_map = {}
        for i_row, r in enumerate(self._rows):
            for i_col, cell in enumerate(r.cells):
                self._row_index_map[id(cell)] = i_row
                self._col_index_map[id(cell)] = i_col
                cell.set_context(r)

        n_cols = set(len(r) for r in self._rows)
        assert len(n_cols) == 1
        self._n_cols: int = n_cols.pop()
        assert self._n_cols > 0

        self._render_col: Tuple[bool, ...] = tuple(
            any(c.should_render == ShouldRender.YES for c in col)
            for col in zip(*[r.cells for r in self._rows])
        )

    def col_reduce(self, caller: Cell, f: Callable[[Tuple["Cell", ...]], Any]) -> Tuple[Any, int]:
        caller_id = id(caller)
        column_index = self._col_index_map[caller_id]
        key = (column_index, id(f))
        if key not in self._reduction_cache:
            self._reduction_cache[key] = f(tuple(r.cells[column_index] for r in self._rows))

        return self._reduction_cache[key], self._row_index_map[caller_id]

    @property
    def col_separator(self) -> str:
        return "  ┊  "

    @staticmethod
    def pad(s: str, width: int, alignment: Alignment) -> str:
        if alignment == Alignment.LEFT:
            return s.ljust(width)
        elif alignment == Alignment.RIGHT:
            return s.rjust(width)
        else:
            assert alignment == Alignment.CENTER
            return s.center(width)

    def render(self) -> None:
        segments: Tuple[Tuple[Tuple[str, ...], ...], ...] = tuple(
            r.render() for r in self._rows)

        alignments: List[Alignment] = [Alignment.ANY for _ in range(self._n_cols)]
        for r in self._rows:
            for i, cell in enumerate(r.cells):
                if alignments[i] == Alignment.ANY:
                    alignments[i] = cell.alignment()
                    continue

                assert cell.alignment() in (alignments[i], Alignment.ANY)

        alignments = [Alignment.CENTER if a == Alignment.ANY else a for a in alignments]
        col_widths: Tuple[int, ...] = tuple(
            max(len(s) for s in it.chain(*col_segments))
            for col_segments in zip(*segments))

        rendered_rows: List[str] = []
        for row_segments in segments:
            padded_row: List[Tuple[str, ...]] = []
            for j in zip(row_segments, col_widths, alignments, self._render_col):
                col, width, alignment, render_col = j
                if render_col:
                    padded_row.append(tuple(self.pad(s, width, alignment) for s in col))

            rendered_rows.append("\n".join([
                self.col_separator.join(ri) for ri in zip(*padded_row)
            ]))

            # Debug only
            print(rendered_rows[-1])


class Null_Cell(Cell):
    def render(self) -> str:
        return ""

    def alignment(self) -> Alignment:
        return Alignment.ANY

    @property
    def should_render(self) -> ShouldRender:
        return ShouldRender.MAYBE


class ColHeader_Cell(Cell):
    def __init__(self, header: str) -> None:
        self._header = header

    def render(self) -> str:
        return self._header

    def alignment(self) -> Alignment:
        return Alignment.ANY

    @property
    def should_render(self) -> ShouldRender:
        return ShouldRender.MAYBE


class Label_Cell(Cell):
    def __init__(self, label: Label) -> None:
        self._label: Label = label

    def render(self) -> str:
        masks, i = self.col_reduce(Label_Cell.gather_masks)
        mask = masks[i]
        assert len(mask) == len(self._label)
        result = "\n".join([
            "  " * i + li
            for i, (mi, li) in enumerate(zip(mask, self._label))
            if mi
        ]) or ""

        # Extra space for top level labels.
        if mask[0]:
            result = f"\n{result}"

        return result

    def alignment(self) -> Alignment:
        return Alignment.LEFT

    @staticmethod
    def gather_masks(col: Tuple[Cell, ...]) -> Tuple[Tuple[bool, ...], ...]:
        masks: List[Tuple[bool, ...]] = []
        prior_l: Label = ()

        for c in col:
            if not isinstance(c, Label_Cell):
                masks.append(())
                continue

            l: Label = c._label
            i: int = 0
            for i, (l_i, pl_i) in enumerate(zip(l, prior_l)):
                if l_i != pl_i:
                    break
                i += 1
            prior_l = l
            masks.append((False,) * i + (True,) * (len(l) - i))

        return tuple(masks)

class NumThreads_Cell(Cell):
    def __init__(self, num_threads: int) -> None:
        self._num_threads: int = num_threads

    def render(self) -> str:
        return f"{self._num_threads}  "

    def alignment(self) -> Alignment:
        return Alignment.RIGHT


class AB_Cell(Cell):
    def __init__(self, a: ValueType, b: ValueType) -> None:
        self._a_counts = int(a[0].counts(denoise=True) / a[0].number_per_run)
        self._b_counts = int(b[0].counts(denoise=True) / b[0].number_per_run)

    def render(self) -> str:
        if self._a_counts == self._b_counts:
            return "..."

        segment_lengths, _ = self.col_reduce(AB_Cell.segment_lengths)
        s0, s1, s2 = [s.rjust(l) for s, l in zip(self.segments, segment_lengths)]
        return f"{s0} -> {s1}  ({s2})"

    @property
    def segments(self) -> Tuple[str, str, str]:
        abs_delta = abs(self._b_counts - self._a_counts)
        rel_delta = abs_delta / statistics.mean([self._a_counts, self._b_counts])

        sign: str = ""
        if self._b_counts < self._a_counts:
            sign = "-"

        elif self._b_counts > self._a_counts:
            sign = "+"

        return (
            f"{sign}{abs_delta}",
            str(self._b_counts),
            f"{sign}{rel_delta * 100:.1f} %"
        )

    @staticmethod
    def segment_lengths(col: Tuple[Cell, ...]) -> Tuple[int, int, int]:
        lengths = [0, 0, 0]
        for c in col:
            if isinstance(c, AB_Cell):
                lengths = [max(li, len(si)) for li, si in zip(lengths, c.segments)]

        assert len(lengths) == 3
        return cast(Tuple[int, int, int], tuple(lengths))

    def alignment(self) -> Alignment:
        return Alignment.RIGHT


def render_ab(a_results: ResultType, b_results: ResultType) -> None:
    packed_results: Dict[
        Tuple[Label, int],
        Dict[Mode, Tuple[ValueType, ValueType]]
    ] = {}

    assert len(a_results) == len(b_results)
    for a_result, b_result in zip(a_results, b_results):
        assert a_result[:3] == b_result[:3]

        label, num_threads, mode, a_value = a_result
        b_value = b_result[3]

        packed_results.setdefault((label, num_threads), {})
        assert mode not in packed_results[(label, num_threads)]
        packed_results[(label, num_threads)][mode] = (a_value, b_value)

    rows: List[Tuple[Cell, ...]] = [
        (Null_Cell(), ColHeader_Cell(header="num  \nthreads")) +
        tuple(ColHeader_Cell(header=m.value) for m in Mode)
    ]

    for (label, num_threads), r in packed_results.items():
        rows.append((
            Label_Cell(label=label),
            NumThreads_Cell(num_threads=num_threads),
        ) + tuple(
            AB_Cell(*r[mode]) if mode in r else Null_Cell()
            for mode in Mode
        ))

    table = Table(rows)
    table.render()
