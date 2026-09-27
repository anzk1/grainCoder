from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BBox = tuple[int, int, int, int]
Axis = Literal["x", "y"]
ElementKind = Literal["text", "non_text"]
NodeType = Literal["row", "column", "atomic"]


@dataclass(frozen=True)
class ElementBox:
    element_id: int
    bbox: BBox
    kind: ElementKind
    confidence: float | None = None
    source: str = "uied"

    @property
    def area(self) -> int:
        left, top, right, bottom = self.bbox
        return max(0, right - left) * max(0, bottom - top)


@dataclass(frozen=True)
class ProjectionGap:
    axis: Axis
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def center(self) -> int:
        return round((self.start + self.end) / 2)


@dataclass
class LayoutNode:
    bbox: BBox
    node_type: NodeType
    children: list["LayoutNode"] = field(default_factory=list)
    split_axis: Axis | None = None
    split_position: int | None = None
    selected_gap: ProjectionGap | None = None
    depth: int = 0


@dataclass(frozen=True)
class NormalizationOperation:
    operation_type: str
    input_ids: tuple[int, ...]
    output_bbox: BBox


@dataclass(frozen=True)
class NormalizationResult:
    boxes: list[ElementBox]
    input_count: int
    operations: list[NormalizationOperation]


@dataclass(frozen=True)
class SplitTraceEntry:
    depth: int
    region_bbox: BBox
    selected_gap: ProjectionGap | None
    outcome: str


@dataclass(frozen=True)
class TreeBuildResult:
    root: LayoutNode
    trace: list[SplitTraceEntry]
    reached_max_depth: bool
    reached_max_leaves: bool
