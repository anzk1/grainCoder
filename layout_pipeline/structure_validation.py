from __future__ import annotations

from dataclasses import dataclass

from .layout_grouping import is_semantic_layout_group
from .schema import ElementNode, LayoutGroup, LayoutNode, RegionProfile, SplitCandidate, atomic_nodes, bbox_area, intersection_area


@dataclass(frozen=True)
class StructureValidation:
    passed: bool
    errors: tuple[str, ...]
    atomic_count: int
    owned_element_count: int
    largest_atomic_element_ratio: float
    thin_text_atomic_ratio: float
    single_element_atomic_ratio: float
    weak_separator_ratio: float
    empty_atomic_ratio: float
    small_single_element_ratio: float
    broken_layout_group_ratio: float
    large_component_cut_ratio: float
    region_type_mismatch_ratio: float

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "atomic_count": self.atomic_count,
            "owned_element_count": self.owned_element_count,
            "largest_atomic_element_ratio": self.largest_atomic_element_ratio,
            "thin_text_atomic_ratio": self.thin_text_atomic_ratio,
            "single_element_atomic_ratio": self.single_element_atomic_ratio,
            "weak_separator_ratio": self.weak_separator_ratio,
            "empty_atomic_ratio": self.empty_atomic_ratio,
            "small_single_element_ratio": self.small_single_element_ratio,
            "broken_layout_group_ratio": self.broken_layout_group_ratio,
            "large_component_cut_ratio": self.large_component_cut_ratio,
            "region_type_mismatch_ratio": self.region_type_mismatch_ratio,
        }


def validate_structure(
    root: LayoutNode,
    elements: list[ElementNode],
    split_candidates: tuple[SplitCandidate, ...] | list[SplitCandidate] = (),
    strong_separator_score: float = 0.9,
    layouts: tuple[LayoutGroup, ...] | list[LayoutGroup] = (),
    region_profiles: tuple[RegionProfile, ...] | list[RegionProfile] = (),
    quality_gate: dict | None = None,
) -> StructureValidation:
    errors: list[str] = []
    valid_types = {"row", "column", "atomic", "spacer"}

    def visit(node: LayoutNode) -> None:
        if node.layout_type not in valid_types:
            errors.append(f"node {node.id} has unsupported type {node.layout_type}")
        if node.layout_type == "atomic" and node.children:
            errors.append(f"atomic node {node.id} has children")
        if node.layout_type == "spacer" and (node.children or node.element_ids):
            errors.append(f"spacer node {node.id} contains structure or elements")
        for child in node.children:
            if not (
                node.bbox[0] <= child.bbox[0]
                and node.bbox[1] <= child.bbox[1]
                and node.bbox[2] >= child.bbox[2]
                and node.bbox[3] >= child.bbox[3]
            ):
                errors.append(f"child {child.id} escapes parent {node.id}")
            visit(child)
        for index, first in enumerate(node.children):
            for second in node.children[index + 1 :]:
                if intersection_area(first.bbox, second.bbox) > 0:
                    errors.append(f"siblings {first.id} and {second.id} overlap")
            if node.layout_type == "row" and index and node.children[index - 1].bbox[0] > first.bbox[0]:
                errors.append(f"row node {node.id} children are out of order")
            if node.layout_type == "column" and index and node.children[index - 1].bbox[1] > first.bbox[1]:
                errors.append(f"column node {node.id} children are out of order")

    visit(root)
    atomics = atomic_nodes(root)
    nonempty_atomics = [atomic for atomic in atomics if atomic.element_ids]
    page_area = bbox_area(root.bbox)
    page_scale = [atomic for atomic in atomics if atomic.area >= page_area * 0.5]
    if len(page_scale) > 1:
        errors.append("multiple atomic regions cover at least half of the page")

    owned_ids = [element_id for atomic in atomics for element_id in atomic.element_ids]
    if len(owned_ids) != len(set(owned_ids)):
        errors.append("one or more UIED elements have duplicate atomic ownership")
    expected_ids = {element.id for element in elements}
    if set(owned_ids) != expected_ids:
        errors.append("atomic ownership does not cover every merged UIED element")

    element_by_id = {element.id: element for element in elements}
    for atomic in atomics:
        owned = [element_by_id[element_id] for element_id in atomic.element_ids]
        if owned and not any(element.is_text for element in owned) and all(
            element.width <= 48 and element.height <= 48 for element in owned
        ):
            errors.append(f"small icon-only atomic region remains: {atomic.id}")

    labels = [atomic.ownership_label for atomic in page_scale if atomic.ownership_label]
    if len(labels) != len(set(labels)):
        errors.append("page-scale semantic ownership is duplicated")

    element_count = max(1, len(elements))
    largest_atomic_element_ratio = max(
        (len(atomic.element_ids) / element_count for atomic in nonempty_atomics),
        default=0.0,
    )
    thin_height = max(96, round(root.height * 0.035))
    thin_text_count = 0
    small_single_count = 0
    for atomic in nonempty_atomics:
        atomic_elements = [element_by_id[element_id] for element_id in atomic.element_ids]
        owned_height = max(element.bbox[3] for element in atomic_elements) - min(
            element.bbox[1] for element in atomic_elements
        )
        if owned_height <= thin_height and all(element.is_text for element in atomic_elements):
            thin_text_count += 1
        if len(atomic_elements) == 1 and _is_bad_small_single(atomic_elements[0], thin_height, page_area):
            small_single_count += 1
    atomic_denominator = max(1, len(nonempty_atomics))
    all_atomic_denominator = max(1, len(atomics))
    single_element_atomic_ratio = sum(len(atomic.element_ids) == 1 for atomic in nonempty_atomics) / atomic_denominator
    selected_candidates = [candidate for candidate in split_candidates if candidate.selected]
    weak_separator_ratio = sum(
        candidate.score < strong_separator_score for candidate in selected_candidates
    ) / max(1, len(selected_candidates))
    empty_atomic_ratio = sum(not atomic.element_ids for atomic in atomics) / all_atomic_denominator
    broken_group_ids = {
        group_id for candidate in selected_candidates for group_id in candidate.layout_group_crossing_ids
    }
    semantic_layouts = [layout for layout in layouts if is_semantic_layout_group(layout)]
    broken_layout_group_ratio = len(broken_group_ids) / max(1, len(semantic_layouts))
    large_component_ids = {
        element_id for candidate in selected_candidates for element_id in candidate.large_component_crossing_ids
    }
    large_elements = [element for element in elements if element.area / max(1, page_area) >= 0.04]
    large_component_cut_ratio = len(large_component_ids) / max(1, len(large_elements))
    accepted_vertical_component_boundary_regions = {
        candidate.region
        for candidate in split_candidates
        if candidate.accepted
        and candidate.line.direction == "vertical"
        and "component_boundary" in candidate.sources
    }
    profile_by_bbox = {profile.features.bbox: profile for profile in region_profiles}
    mixed_atomic_threshold = float((quality_gate or {}).get("mixed_atomic_element_ratio", 0.75))
    for atomic in nonempty_atomics:
        profile = profile_by_bbox.get(atomic.bbox)
        if profile is None:
            continue
        features = profile.features
        owned_elements = [element_by_id[element_id] for element_id in atomic.element_ids]
        if (
            profile.region_type == "two_column"
            and features.foreground_count >= 4
            and max(features.strongest_vertical_gap, features.strongest_vertical_separator) >= 0.15
        ):
            errors.append(f"two-column atomic {atomic.id} remains unsplit")
            continue
        if (
            profile.region_type in {"text_section", "mixed"}
            and features.text_count >= 2
            and any(element.segmentation_role != "foreground" for element in owned_elements)
            and atomic.bbox in accepted_vertical_component_boundary_regions
        ):
            errors.append(
                f"heterogeneous atomic {atomic.id} remains unsplit despite accepted vertical component boundary"
            )
            continue
        if profile.region_type != "mixed":
            continue
        internal_boundary = max(
            features.strongest_horizontal_gap,
            features.strongest_vertical_gap,
            features.strongest_horizontal_separator,
            features.strongest_vertical_separator,
            features.horizontal_color_transition,
            features.vertical_color_transition,
        )
        group_count = features.row_group_count + features.column_group_count + features.grid_group_count
        element_ratio = len(atomic.element_ids) / element_count
        if element_ratio > mixed_atomic_threshold and (internal_boundary >= 0.45 or group_count >= 2):
            errors.append(
                f"mixed atomic {atomic.id} owns {element_ratio:.6f} of elements despite internal partition evidence"
            )
    mismatch_count = 0
    specialized_count = 0
    for candidate in selected_candidates:
        profile = profile_by_bbox.get(candidate.region)
        if profile is None or profile.region_type not in {"hero", "two_column"}:
            continue
        specialized_count += 1
        if profile.region_type == "hero" and candidate.score < strong_separator_score:
            mismatch_count += 1
            errors.append(f"hero region {candidate.region} is fragmented by a weak separator")
        if profile.region_type == "two_column" and candidate.line.direction != "vertical":
            mismatch_count += 1
            errors.append(f"two-column region {candidate.region} uses a horizontal first partition")
    region_type_mismatch_ratio = mismatch_count / max(1, specialized_count)

    ratios = {
        "empty_atomic_ratio": empty_atomic_ratio,
        "thin_text_atomic_ratio": thin_text_count / atomic_denominator,
        "small_single_element_ratio": small_single_count / atomic_denominator,
        "broken_layout_group_ratio": broken_layout_group_ratio,
        "large_component_cut_ratio": large_component_cut_ratio,
        "region_type_mismatch_ratio": region_type_mismatch_ratio,
    }
    thresholds = {
        "empty_atomic_ratio": 0.0,
        "thin_text_atomic_ratio": 0.40,
        "small_single_element_ratio": 0.45,
        "broken_layout_group_ratio": 0.10,
        "large_component_cut_ratio": 0.10,
        "region_type_mismatch_ratio": 0.10,
        **(quality_gate or {}),
    }
    for name, ratio in ratios.items():
        if ratio > float(thresholds[name]):
            errors.append(f"{name} {ratio:.6f} exceeds {float(thresholds[name]):.6f}")

    return StructureValidation(
        not errors,
        tuple(dict.fromkeys(errors)),
        len(nonempty_atomics),
        len(set(owned_ids)),
        round(largest_atomic_element_ratio, 6),
        round(thin_text_count / atomic_denominator, 6),
        round(single_element_atomic_ratio, 6),
        round(weak_separator_ratio, 6),
        round(empty_atomic_ratio, 6),
        round(small_single_count / atomic_denominator, 6),
        round(broken_layout_group_ratio, 6),
        round(large_component_cut_ratio, 6),
        round(region_type_mismatch_ratio, 6),
    )


def _is_bad_small_single(element: ElementNode, thin_height: int, page_area: int) -> bool:
    if element.is_text:
        return element.height <= thin_height and element.area / max(1, page_area) < 0.03
    if element.area / max(1, page_area) >= 0.04:
        return False
    return element.width <= 48 and element.height <= 48
