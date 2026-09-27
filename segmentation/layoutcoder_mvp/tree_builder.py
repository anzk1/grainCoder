from __future__ import annotations

from .projection import boxes_intersect_region, find_projection_gaps
from .types import BBox, ElementBox, LayoutNode, ProjectionGap, SplitTraceEntry, TreeBuildResult


class ProjectionTreeBuilder:
    def __init__(
        self,
        max_depth: int = 2,
        max_leaves: int = 20,
        min_gap_px: int = 8,
        min_child_width_px: int = 24,
        min_child_height_px: int = 16,
    ):
        if max_depth < 0 or max_leaves < 1:
            raise ValueError("max_depth must be non-negative and max_leaves must be positive")
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_gap_px = min_gap_px
        self.min_child_width_px = min_child_width_px
        self.min_child_height_px = min_child_height_px
        self._leaf_count = 1
        self._trace: list[SplitTraceEntry] = []
        self._reached_max_depth = False
        self._reached_max_leaves = False

    def build(self, image_size: tuple[int, int], boxes: list[ElementBox]) -> TreeBuildResult:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        self._leaf_count = 1
        self._trace = []
        self._reached_max_depth = False
        self._reached_max_leaves = False
        root = self._parse_region((0, 0, width, height), boxes, 0)
        return TreeBuildResult(root, list(self._trace), self._reached_max_depth, self._reached_max_leaves)

    def _parse_region(self, region: BBox, boxes: list[ElementBox], depth: int) -> LayoutNode:
        region_boxes = [box for box in boxes if boxes_intersect_region(box, region)]
        if depth >= self.max_depth:
            self._reached_max_depth = self._reached_max_depth or bool(find_projection_gaps(region_boxes, region, self.min_gap_px))
            self._trace.append(SplitTraceEntry(depth, region, None, "max_depth"))
            return LayoutNode(region, "atomic", depth=depth)
        if self._leaf_count >= self.max_leaves:
            self._reached_max_leaves = True
            self._trace.append(SplitTraceEntry(depth, region, None, "max_leaves"))
            return LayoutNode(region, "atomic", depth=depth)

        for gap in find_projection_gaps(region_boxes, region, self.min_gap_px):
            children = self._split_region(region, gap)
            if children is None or not self._valid_children(children, region_boxes, gap):
                self._trace.append(SplitTraceEntry(depth, region, gap, "rejected"))
                continue
            self._leaf_count += 1
            first_boxes = [box for box in region_boxes if boxes_intersect_region(box, children[0])]
            second_boxes = [box for box in region_boxes if boxes_intersect_region(box, children[1])]
            node_type = "row" if gap.axis == "x" else "column"
            node = LayoutNode(
                region,
                node_type,
                split_axis=gap.axis,
                split_position=gap.center,
                selected_gap=gap,
                depth=depth,
            )
            self._trace.append(SplitTraceEntry(depth, region, gap, "selected"))
            node.children = [
                self._parse_region(children[0], first_boxes, depth + 1),
                self._parse_region(children[1], second_boxes, depth + 1),
            ]
            return node

        self._trace.append(SplitTraceEntry(depth, region, None, "atomic"))
        return LayoutNode(region, "atomic", depth=depth)

    def _split_region(self, region: BBox, gap: ProjectionGap) -> tuple[BBox, BBox] | None:
        left, top, right, bottom = region
        position = gap.center
        if gap.axis == "x":
            if position <= left or position >= right:
                return None
            return (left, top, position, bottom), (position, top, right, bottom)
        if position <= top or position >= bottom:
            return None
        return (left, top, right, position), (left, position, right, bottom)

    def _valid_children(
        self, children: tuple[BBox, BBox], boxes: list[ElementBox], gap: ProjectionGap
    ) -> bool:
        if self._leaf_count + 1 > self.max_leaves:
            self._reached_max_leaves = True
            return False
        for child in children:
            if child[2] - child[0] < self.min_child_width_px:
                return False
            if child[3] - child[1] < self.min_child_height_px:
                return False
        first_count = sum(boxes_intersect_region(box, children[0]) for box in boxes)
        second_count = sum(boxes_intersect_region(box, children[1]) for box in boxes)
        if first_count == 0 or second_count == 0:
            return False
        position = gap.center
        for box in boxes:
            if gap.axis == "x" and box.bbox[0] < position < box.bbox[2]:
                return False
            if gap.axis == "y" and box.bbox[1] < position < box.bbox[3]:
                return False
        return True
