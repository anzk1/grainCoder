from __future__ import annotations

from dataclasses import replace

from .schema import ElementNode, LayoutNode, RegionProfile, union_bbox


def merge_weak_text_atomics(
    root: LayoutNode,
    elements: list[ElementNode],
    strong_separator_score: float,
    region_profiles: tuple[RegionProfile, ...] | list[RegionProfile] = (),
) -> LayoutNode:
    element_by_id = {element.id: element for element in elements}
    profile_by_bbox = {profile.features.bbox: profile for profile in region_profiles}

    def merge_node(node: LayoutNode) -> LayoutNode:
        if node.layout_type == "atomic":
            return node
        children = tuple(merge_node(child) for child in node.children)
        children = _flatten_weak_same_axis(children, node.layout_type, strong_separator_score)
        rebuilt = replace(node, children=children)
        if rebuilt.split_score is None or rebuilt.split_score >= strong_separator_score:
            return rebuilt
        if "lsd" in rebuilt.split_sources or "color_change" in rebuilt.split_sources:
            return rebuilt

        merged_children: list[LayoutNode] = []
        run: list[LayoutNode] = []
        for child in children:
            if child.layout_type != "atomic":
                if run:
                    merged_children.append(_merge_atomic_run(run))
                    run = []
                merged_children.append(child)
                continue
            if not run or _can_join_run(run, child, element_by_id, profile_by_bbox):
                run.append(child)
            else:
                merged_children.append(_merge_atomic_run(run))
                run = [child]
        if run:
            merged_children.append(_merge_atomic_run(run))

        if len(merged_children) == 1 and merged_children[0].layout_type == "atomic":
            merged = merged_children[0]
            return replace(merged, id=rebuilt.id, bbox=rebuilt.bbox, portion=rebuilt.portion)
        return replace(rebuilt, children=tuple(merged_children))

    return merge_node(root)


def _flatten_weak_same_axis(
    children: tuple[LayoutNode, ...],
    layout_type: str,
    strong_separator_score: float,
) -> tuple[LayoutNode, ...]:
    flattened = []
    for child in children:
        if (
            child.layout_type == layout_type
            and child.split_score is not None
            and child.split_score < strong_separator_score
            and "lsd" not in child.split_sources
            and "color_change" not in child.split_sources
        ):
            flattened.extend(child.children)
        else:
            flattened.append(child)
    return tuple(flattened)


def _can_join_run(
    run: list[LayoutNode],
    child: LayoutNode,
    element_by_id: dict[int, ElementNode],
    profile_by_bbox: dict,
) -> bool:
    first = run[0]
    if not first.element_ids or not child.element_ids:
        return False
    if first.ownership_label != child.ownership_label:
        return False
    if first.region_type != child.region_type:
        return False
    if first.region_type == "card_grid":
        return False
    run_elements = [element_by_id[element_id] for node in run for element_id in node.element_ids]
    child_elements = [element_by_id[element_id] for element_id in child.element_ids]
    if not all(_mergeable_element(element) for element in run_elements + child_elements):
        return False
    profiles = [profile_by_bbox.get(node.bbox) for node in run + [child]]
    known_profiles = [profile for profile in profiles if profile is not None]
    return not known_profiles or all(profile.features.background_continuity >= 0.65 for profile in known_profiles)


def _mergeable_element(element: ElementNode) -> bool:
    if element.is_text:
        return True
    if element.width <= 64 and element.height <= 64:
        return True
    return element.segmentation_role != "foreground" and element.area <= 4096


def _merge_atomic_run(run: list[LayoutNode]) -> LayoutNode:
    if len(run) == 1:
        return run[0]
    return LayoutNode(
        run[0].id,
        union_bbox(tuple(node.bbox for node in run)),
        "atomic",
        portion=sum(node.portion for node in run),
        element_ids=tuple(sorted(element_id for node in run for element_id in node.element_ids)),
        ownership_label=run[0].ownership_label,
        region_type=run[0].region_type,
    )
