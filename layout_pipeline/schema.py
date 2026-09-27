from __future__ import annotations

from dataclasses import dataclass, replace
from math import gcd
from typing import Literal

BBox = tuple[int, int, int, int]
LayoutType = Literal["row", "column", "atomic", "spacer"]
ElementRole = Literal["foreground", "container", "background"]
SeparatorSource = Literal["lsd", "whitespace", "layout_group", "component_boundary", "color_change"]
PagePosition = Literal["header", "content", "footer"]
RegionType = Literal[
    "hero",
    "navigation",
    "text_section",
    "card_grid",
    "two_column",
    "list",
    "footer",
    "mixed",
]


def intersection_area(first: BBox, second: BBox) -> int:
    return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )


def bbox_area(bbox: BBox) -> int:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def union_bbox(boxes: list[BBox] | tuple[BBox, ...]) -> BBox:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def numbers_to_portions(lengths: list[int]) -> list[int]:
    divisor = 0
    for length in lengths:
        divisor = gcd(divisor, max(1, round(length)))
    return [max(1, round(length)) // max(1, divisor) for length in lengths]


@dataclass(frozen=True)
class ElementNode:
    id: int
    bbox: BBox
    uied_class: str
    text: str | None = None
    segmentation_role: ElementRole = "foreground"

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return bbox_area(self.bbox)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def is_text(self) -> bool:
        return self.uied_class == "Text"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": list(self.bbox),
            "class": self.uied_class,
            "text": self.text,
            "segmentation_role": self.segmentation_role,
        }


@dataclass(frozen=True)
class LayoutGroup:
    group_id: int
    layout_type: Literal["row", "column", "grid"]
    bbox: BBox
    element_ids: tuple[int, ...]
    uied_class: str

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "layout_type": self.layout_type,
            "bbox": list(self.bbox),
            "element_ids": list(self.element_ids),
            "class": self.uied_class,
        }


@dataclass(frozen=True)
class RegionFeatures:
    bbox: BBox
    page_width_ratio: float
    page_height_ratio: float
    page_area_ratio: float
    page_position: PagePosition
    foreground_count: int
    text_count: int
    non_text_count: int
    text_ratio: float
    element_density: float
    median_text_height: float
    text_line_count: int
    horizontal_text_spread: float
    vertical_text_spread: float
    largest_component_area_ratio: float
    container_coverage_ratio: float
    row_group_count: int
    column_group_count: int
    grid_group_count: int
    repeated_size_score: float
    repeated_gap_score: float
    strongest_horizontal_gap: float
    strongest_vertical_gap: float
    strongest_horizontal_separator: float
    strongest_vertical_separator: float
    color_variance: float
    horizontal_color_transition: float
    vertical_color_transition: float
    background_continuity: float
    edge_density: float

    def to_dict(self) -> dict:
        return {
            field: list(value) if field == "bbox" else value
            for field, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class RegionProfile:
    region_type: RegionType
    confidence: float
    features: RegionFeatures
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "region_type": self.region_type,
            "confidence": self.confidence,
            "features": self.features.to_dict(),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SeparatorLine:
    direction: Literal["horizontal", "vertical"]
    position: int
    start: int
    end: int
    source: SeparatorSource

    def to_dict(self) -> dict:
        if self.direction == "horizontal":
            coordinates = {"x1": self.start, "y1": self.position, "x2": self.end, "y2": self.position}
        else:
            coordinates = {"x1": self.position, "y1": self.start, "x2": self.position, "y2": self.end}
        return {**coordinates, "direction": self.direction, "source": self.source}


@dataclass(frozen=True)
class SplitCandidate:
    region: BBox
    line: SeparatorLine
    sources: tuple[SeparatorSource, ...]
    score: float
    accepted: bool
    rejection_reasons: tuple[str, ...]
    foreground_before: int
    foreground_after: int
    foreground_crossing_ids: tuple[int, ...]
    nonforeground_crossing_ids: tuple[int, ...]
    gap_width: int
    child_balance: float
    color_strength: float = 0.0
    selected: bool = False
    layout_group_crossing_ids: tuple[int, ...] = ()
    large_component_crossing_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {
            "region": list(self.region),
            "line": self.line.to_dict(),
            "sources": list(self.sources),
            "score": self.score,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "foreground_before": self.foreground_before,
            "foreground_after": self.foreground_after,
            "foreground_crossing_ids": list(self.foreground_crossing_ids),
            "nonforeground_crossing_ids": list(self.nonforeground_crossing_ids),
            "gap_width": self.gap_width,
            "child_balance": self.child_balance,
            "color_strength": self.color_strength,
            "selected": self.selected,
            "layout_group_crossing_ids": list(self.layout_group_crossing_ids),
            "large_component_crossing_ids": list(self.large_component_crossing_ids),
        }


@dataclass(frozen=True)
class RegionDecisionCandidate:
    decision_type: Literal["no_split", "split"]
    split_candidate: SplitCandidate | None
    local_score: float
    policy_adjustment: float
    final_local_score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "decision_type": self.decision_type,
            "split_candidate": self.split_candidate.to_dict() if self.split_candidate is not None else None,
            "local_score": self.local_score,
            "policy_adjustment": self.policy_adjustment,
            "final_local_score": self.final_local_score,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SplitNode:
    bbox: BBox
    children: tuple["SplitNode", ...] = ()
    split_line: SeparatorLine | None = None
    split_score: float | None = None
    split_sources: tuple[SeparatorSource, ...] = ()
    region_type: RegionType | None = None

    def to_dict(self) -> dict:
        payload = {"bbox": list(self.bbox)}
        if self.region_type is not None:
            payload["region_type"] = self.region_type
        if self.split_line is not None:
            payload["split_line"] = self.split_line.to_dict()
            payload["split_score"] = self.split_score
            payload["split_sources"] = list(self.split_sources)
        if self.children:
            payload["children"] = [child.to_dict() for child in self.children]
        return payload


@dataclass(frozen=True)
class LayoutNode:
    id: int
    bbox: BBox
    layout_type: LayoutType
    children: tuple["LayoutNode", ...] = ()
    portion: int = 1
    element_ids: tuple[int, ...] = ()
    ownership_label: str | None = None
    split_score: float | None = None
    split_sources: tuple[SeparatorSource, ...] = ()
    region_type: RegionType | None = None

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return bbox_area(self.bbox)

    def with_children(self, children: tuple["LayoutNode", ...]) -> "LayoutNode":
        return replace(self, children=children)

    def to_dict(self) -> dict:
        payload = {
            "id": self.id,
            "type": self.layout_type,
            "bbox": list(self.bbox),
            "portion": self.portion,
            "element_ids": list(self.element_ids),
        }
        if self.ownership_label is not None:
            payload["ownership_label"] = self.ownership_label
        if self.split_score is not None:
            payload["split_score"] = self.split_score
            payload["split_sources"] = list(self.split_sources)
        if self.region_type is not None:
            payload["region_type"] = self.region_type
        if self.children:
            payload["children"] = [child.to_dict() for child in self.children]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "LayoutNode":
        layout_type = payload["type"]
        if layout_type not in {"row", "column", "atomic", "spacer"}:
            raise ValueError(f"unsupported layout type: {layout_type}")
        bbox = tuple(round(value) for value in payload["bbox"])
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("layout bbox must have positive area")
        return cls(
            id=int(payload["id"]),
            bbox=bbox,
            layout_type=layout_type,
            children=tuple(cls.from_dict(child) for child in payload.get("children", [])),
            portion=max(1, int(payload.get("portion", 1))),
            element_ids=tuple(int(element_id) for element_id in payload.get("element_ids", [])),
            ownership_label=payload.get("ownership_label"),
            split_score=float(payload["split_score"]) if payload.get("split_score") is not None else None,
            split_sources=tuple(payload.get("split_sources", [])),
            region_type=payload.get("region_type"),
        )


def atomic_nodes(root: LayoutNode) -> tuple[LayoutNode, ...]:
    if root.layout_type == "atomic":
        return (root,)
    return tuple(atomic for child in root.children for atomic in atomic_nodes(child))
