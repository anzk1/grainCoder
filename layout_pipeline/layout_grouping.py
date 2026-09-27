from __future__ import annotations

from collections import deque

from .schema import BBox, ElementNode, LayoutGroup, SeparatorLine, union_bbox


class Box:
    def __init__(self, element: ElementNode):
        self.element = element
        self.box_id = element.id
        self.column_min, self.row_min, self.column_max, self.row_max = element.bbox
        self.width = element.width
        self.height = element.height
        self.left_children: list[int] = []
        self.right_children: list[int] = []
        self.top_children: list[int] = []
        self.bottom_children: list[int] = []
        self.is_left_aligned = False
        self.is_right_aligned = False
        self.is_top_aligned = False
        self.is_bottom_aligned = False
        self.left_gap: dict[int, int] = {}
        self.right_gap: dict[int, int] = {}
        self.top_gap: dict[int, int] = {}
        self.bottom_gap: dict[int, int] = {}


def _is_adjacent(first: Box, second: Box, direction: str) -> bool:
    if direction == "left":
        return first.column_max <= second.column_min and first.row_max > second.row_min and first.row_min < second.row_max
    if direction == "right":
        return first.column_min >= second.column_max and first.row_max > second.row_min and first.row_min < second.row_max
    if direction == "top":
        return first.row_max <= second.row_min and first.column_max > second.column_min and first.column_min < second.column_max
    return first.row_min >= second.row_max and first.column_max > second.column_min and first.column_min < second.column_max


def _is_blocked(first: Box, second: Box, boxes: list[Box], direction: str) -> bool:
    for blocker in boxes:
        if blocker.box_id in {first.box_id, second.box_id}:
            continue
        if direction in {"left", "right"}:
            overlaps_axis = max(first.row_min, second.row_min) < blocker.row_max and min(first.row_max, second.row_max) > blocker.row_min
            between = first.column_max <= blocker.column_min < second.column_min if direction == "left" else first.column_min >= blocker.column_max > second.column_max
        else:
            overlaps_axis = max(first.column_min, second.column_min) < blocker.column_max and min(first.column_max, second.column_max) > blocker.column_min
            between = first.row_max <= blocker.row_min < second.row_min if direction == "top" else first.row_min >= blocker.row_max > second.row_max
        if overlaps_axis and between:
            return True
    return False


def find_adjacent_boxes(boxes: list[Box]) -> dict[int, Box]:
    indexed = {box.box_id: box for box in boxes}
    child_fields = {
        "left": "right_children",
        "right": "left_children",
        "top": "bottom_children",
        "bottom": "top_children",
    }
    for first in boxes:
        for direction, child_field in child_fields.items():
            candidates = [second for second in boxes if second is not first and _is_adjacent(first, second, direction)]
            setattr(first, child_field, [second.box_id for second in candidates if not _is_blocked(first, second, boxes, direction)])
    return indexed


def compute_box_alignment(boxes: dict[int, Box]) -> None:
    for center in boxes.values():
        for direction, child_ids, attribute in (
            ("right", center.right_children, "is_right_aligned"),
            ("left", center.left_children, "is_left_aligned"),
            ("bottom", center.bottom_children, "is_bottom_aligned"),
            ("top", center.top_children, "is_top_aligned"),
        ):
            if len(child_ids) != 1:
                continue
            child = boxes[child_ids[0]]
            threshold = max(center.width, center.height, child.width, child.height) * 0.03
            if direction in {"left", "right"}:
                aligned = abs(center.row_min - child.row_min) <= threshold and abs(center.row_max - child.row_max) <= threshold
            else:
                aligned = abs(center.column_min - child.column_min) <= threshold and abs(center.column_max - child.column_max) <= threshold
            setattr(center, attribute, aligned)


def compute_box_gap(boxes: dict[int, Box]) -> None:
    for center in boxes.values():
        center.right_gap = {child_id: boxes[child_id].column_min - center.column_max for child_id in center.right_children}
        center.left_gap = {child_id: center.column_min - boxes[child_id].column_max for child_id in center.left_children}
        center.bottom_gap = {child_id: boxes[child_id].row_min - center.row_max for child_id in center.bottom_children}
        center.top_gap = {child_id: center.row_min - boxes[child_id].row_max for child_id in center.top_children}


def _connected_neighbors(box: Box, boxes: dict[int, Box]) -> list[Box]:
    neighbors: list[Box] = []
    for child_ids, aligned, gaps in (
        (box.right_children, box.is_right_aligned, box.right_gap),
        (box.left_children, box.is_left_aligned, box.left_gap),
        (box.bottom_children, box.is_bottom_aligned, box.bottom_gap),
        (box.top_children, box.is_top_aligned, box.top_gap),
    ):
        if not aligned:
            continue
        neighbors.extend(boxes[child_id] for child_id in child_ids if gaps.get(child_id, -1) >= 0)
    return neighbors


def expand_group(start_box: Box, boxes: dict[int, Box], visited: set[int]) -> list[Box]:
    group = [start_box]
    queue = deque([start_box])
    while queue:
        center = queue.popleft()
        for neighbor in _connected_neighbors(center, boxes):
            if neighbor.box_id in visited:
                continue
            visited.add(neighbor.box_id)
            group.append(neighbor)
            queue.append(neighbor)
    return group


def analyze_group(group: list[Box]) -> str | None:
    rows = _cluster_axis(group, "row")
    columns = _cluster_axis(group, "column")
    if len(rows) > 1 and len(columns) > 1 and len(rows) * len(columns) == len(group):
        return "grid"
    if len(rows) == 1 and len(group) > 1:
        return "row"
    if len(columns) == 1 and len(group) > 1:
        return "column"
    return None


def _cluster_axis(group: list[Box], axis: str) -> list[list[Box]]:
    ordered = sorted(group, key=lambda box: (box.row_min, box.column_min) if axis == "row" else (box.column_min, box.row_min))
    clusters: list[list[Box]] = []
    for box in ordered:
        center = (box.row_min + box.row_max) / 2 if axis == "row" else (box.column_min + box.column_max) / 2
        matching = next((cluster for cluster in clusters if _center_inside(cluster[0], center, axis)), None)
        if matching is None:
            clusters.append([box])
        else:
            matching.append(box)
    return clusters


def _center_inside(reference: Box, center: float, axis: str) -> bool:
    if axis == "row":
        return reference.row_min <= center <= reference.row_max
    return reference.column_min <= center <= reference.column_max


def _line_crosses_bbox(line: SeparatorLine, bbox: BBox) -> bool:
    if line.direction == "horizontal":
        return bbox[1] < line.position < bbox[3] and max(bbox[0], line.start) < min(bbox[2], line.end)
    return bbox[0] < line.position < bbox[2] and max(bbox[1], line.start) < min(bbox[3], line.end)


def search_layout(elements: list[ElementNode], uied_class: str, separators: list[SeparatorLine] | None = None) -> list[LayoutGroup]:
    boxes = find_adjacent_boxes([Box(element) for element in elements])
    compute_box_alignment(boxes)
    compute_box_gap(boxes)
    visited: set[int] = set()
    groups: list[LayoutGroup] = []
    for box_id in sorted(boxes):
        if box_id in visited:
            continue
        visited.add(box_id)
        group = expand_group(boxes[box_id], boxes, visited)
        layout_type = analyze_group(group)
        if layout_type is None:
            continue
        bbox = union_bbox([box.element.bbox for box in group])
        if separators and any(_line_crosses_bbox(line, bbox) for line in separators):
            continue
        groups.append(LayoutGroup(len(groups), layout_type, bbox, tuple(sorted(box.box_id for box in group)), uied_class))
    return groups


def build_layout_groups(elements: list[ElementNode], separators: list[SeparatorLine] | None = None) -> list[LayoutGroup]:
    text_elements = [element for element in elements if element.uied_class == "Text"]
    non_text_elements = [element for element in elements if element.uied_class in {"Compo", "Block"}]
    groups = search_layout(text_elements, "Text", separators) + search_layout(non_text_elements, "Compo", separators)
    return [LayoutGroup(index, group.layout_type, group.bbox, group.element_ids, group.uied_class) for index, group in enumerate(groups)]


def is_semantic_layout_group(layout: LayoutGroup) -> bool:
    if layout.uied_class == "Text" or layout.layout_type == "grid":
        return True
    width = layout.bbox[2] - layout.bbox[0]
    height = layout.bbox[3] - layout.bbox[1]
    if layout.layout_type == "column" and width <= 64:
        return False
    if layout.layout_type == "row" and height <= 64:
        return False
    return True
