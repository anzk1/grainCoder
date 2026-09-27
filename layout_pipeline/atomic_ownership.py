from __future__ import annotations

from dataclasses import replace
from math import hypot

from .schema import ElementNode, LayoutNode, atomic_nodes, intersection_area


def _contains_center(node: LayoutNode, element: ElementNode) -> bool:
    center_x, center_y = element.center
    return node.bbox[0] <= center_x < node.bbox[2] and node.bbox[1] <= center_y < node.bbox[3]


def _bbox_distance(first: LayoutNode, second: LayoutNode) -> float:
    first_center = ((first.bbox[0] + first.bbox[2]) / 2, (first.bbox[1] + first.bbox[3]) / 2)
    second_center = ((second.bbox[0] + second.bbox[2]) / 2, (second.bbox[1] + second.bbox[3]) / 2)
    return hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])


def _page_label(node: LayoutNode, page_height: int) -> str:
    center_y = (node.bbox[1] + node.bbox[3]) / 2
    if center_y < page_height * 0.2:
        return "header"
    if center_y > page_height * 0.8:
        return "footer"
    return "content"


def assign_atomic_ownership(
    root: LayoutNode,
    elements: list[ElementNode],
    page_size: tuple[int, int],
    icon_max_size: int = 48,
) -> LayoutNode:
    atomics = list(atomic_nodes(root))
    ownership: dict[int, list[int]] = {atomic.id: [] for atomic in atomics}
    element_by_id = {element.id: element for element in elements}
    for element in elements:
        containing = [atomic for atomic in atomics if _contains_center(atomic, element)]
        if containing:
            owner = min(containing, key=lambda atomic: atomic.area)
        else:
            owner = max(atomics, key=lambda atomic: (intersection_area(atomic.bbox, element.bbox), -atomic.id))
        ownership[owner.id].append(element.id)

    for atomic in atomics:
        owned = [element_by_id[element_id] for element_id in ownership[atomic.id]]
        if not owned or any(element.is_text for element in owned):
            continue
        if not all(element.width <= icon_max_size and element.height <= icon_max_size for element in owned):
            continue
        candidates = []
        for candidate in atomics:
            if candidate.id == atomic.id:
                continue
            candidate_elements = [element_by_id[element_id] for element_id in ownership[candidate.id]]
            if any(element.is_text for element in candidate_elements) or any(
                element.width > icon_max_size or element.height > icon_max_size
                for element in candidate_elements
            ):
                candidates.append(candidate)
        if not candidates:
            continue
        receiver = min(candidates, key=lambda candidate: (_bbox_distance(atomic, candidate), candidate.id))
        ownership[receiver.id].extend(ownership[atomic.id])
        ownership[atomic.id] = []

    def rebuild(node: LayoutNode) -> LayoutNode:
        if node.layout_type == "atomic":
            return replace(
                node,
                element_ids=tuple(sorted(ownership[node.id])),
                ownership_label=_page_label(node, page_size[1]),
            )
        return replace(node, children=tuple(rebuild(child) for child in node.children))

    return rebuild(root)


def replace_empty_atomics_with_spacers(root: LayoutNode) -> LayoutNode:
    if root.layout_type == "atomic" and not root.element_ids:
        return replace(root, layout_type="spacer", ownership_label=None, region_type=None)
    return replace(root, children=tuple(replace_empty_atomics_with_spacers(child) for child in root.children))
