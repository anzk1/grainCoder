from __future__ import annotations

from .schema import LayoutNode, SplitNode, numbers_to_portions


def layout_from_split_tree(split_root: SplitNode) -> LayoutNode:
    next_id = 0

    def convert(split_node: SplitNode, portion: int) -> LayoutNode:
        nonlocal next_id
        node_id = next_id
        next_id += 1
        if not split_node.children or split_node.split_line is None:
            return LayoutNode(node_id, split_node.bbox, "atomic", portion=portion, region_type=split_node.region_type)
        if split_node.split_line.direction == "vertical":
            layout_type = "row"
            lengths = [child.bbox[2] - child.bbox[0] for child in split_node.children]
        else:
            layout_type = "column"
            lengths = [child.bbox[3] - child.bbox[1] for child in split_node.children]
        child_portions = numbers_to_portions(lengths)
        children = tuple(
            convert(child, child_portion)
            for child, child_portion in zip(split_node.children, child_portions)
        )
        return LayoutNode(
            node_id,
            split_node.bbox,
            layout_type,
            children,
            portion,
            split_score=split_node.split_score,
            split_sources=split_node.split_sources,
            region_type=split_node.region_type,
        )

    return convert(split_root, 1)
