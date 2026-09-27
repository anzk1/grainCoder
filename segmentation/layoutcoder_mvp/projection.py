from typing import Literal

from .types import BBox, ElementBox, ProjectionGap


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or merged[-1][1] < start:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def boxes_intersect_region(box: ElementBox, region_bbox: BBox) -> bool:
    return (
        box.bbox[0] < region_bbox[2]
        and box.bbox[2] > region_bbox[0]
        and box.bbox[1] < region_bbox[3]
        and box.bbox[3] > region_bbox[1]
    )


def project_boxes(
    boxes: list[ElementBox], region_bbox: BBox, axis: Literal["x", "y"]
) -> list[tuple[int, int]]:
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    start_index, end_index = (0, 2) if axis == "x" else (1, 3)
    region_start, region_end = region_bbox[start_index], region_bbox[end_index]
    intervals = []
    for box in boxes:
        if boxes_intersect_region(box, region_bbox):
            start = max(box.bbox[start_index], region_start)
            end = min(box.bbox[end_index], region_end)
            if end > start:
                intervals.append((start, end))
    return merge_intervals(intervals)


def find_internal_gaps(
    merged_intervals: list[tuple[int, int]],
    region_start: int,
    region_end: int,
    axis: Literal["x", "y"],
    min_gap_px: int,
) -> list[ProjectionGap]:
    if region_end <= region_start:
        raise ValueError("region end must be greater than region start")
    gaps = []
    clipped = merge_intervals(
        [(max(start, region_start), min(end, region_end)) for start, end in merged_intervals]
    )
    for previous, current in zip(clipped, clipped[1:]):
        if current[0] - previous[1] >= min_gap_px:
            gaps.append(ProjectionGap(axis, previous[1], current[0]))
    return gaps


def find_projection_gaps(
    boxes: list[ElementBox], region_bbox: BBox, min_gap_px: int
) -> list[ProjectionGap]:
    x_intervals = project_boxes(boxes, region_bbox, "x")
    y_intervals = project_boxes(boxes, region_bbox, "y")
    gaps = find_internal_gaps(x_intervals, region_bbox[0], region_bbox[2], "x", min_gap_px)
    gaps.extend(find_internal_gaps(y_intervals, region_bbox[1], region_bbox[3], "y", min_gap_px))
    axis_priority = {"y": 0, "x": 1}
    return sorted(gaps, key=lambda gap: (-gap.size, axis_priority[gap.axis], gap.start, gap.end))
