from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout_grouping import is_semantic_layout_group
from .schema import BBox, ElementNode, LayoutGroup, RegionDecisionCandidate, RegionProfile, SplitNode, intersection_area


@dataclass(frozen=True)
class TreeScoreBreakdown:
    total_score: float
    positive: dict[str, float]
    negative: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "total_score": self.total_score,
            "positive": self.positive,
            "negative": self.negative,
        }


def score_split_tree(
    split_tree: SplitNode,
    decision_traces: tuple,
    elements: list[ElementNode],
    layouts: list[LayoutGroup],
    image_pixels: np.ndarray,
    weights: dict,
) -> TreeScoreBreakdown:
    leaves = _leaf_nodes(split_tree)
    split_traces = [trace for trace in decision_traces if trace.decision.split_candidate is not None]
    profile_by_bbox = {trace.profile.features.bbox: trace.profile for trace in decision_traces}
    leaf_contents = [
        [element for element in elements if intersection_area(element.bbox, leaf.bbox) > 0]
        for leaf in leaves
    ]
    separator_evidence = _mean(
        min(1.0, trace.decision.split_candidate.score)
        for trace in split_traces
    )
    semantic_cohesion = _mean(_profile_cohesion(trace.profile, trace.decision) for trace in decision_traces)
    crossed_group_ids = {
        group_id
        for trace in split_traces
        for group_id in trace.decision.split_candidate.layout_group_crossing_ids
    }
    semantic_layouts = [layout for layout in layouts if is_semantic_layout_group(layout)]
    layout_group_preservation = 1.0 - len(crossed_group_ids) / max(1, len(semantic_layouts))
    repeated_unit_preservation = _mean(
        (trace.profile.features.repeated_size_score + trace.profile.features.repeated_gap_score) / 2
        for trace in decision_traces
        if trace.profile.region_type in {"card_grid", "list"}
    )
    direction_consistency = _mean(
        1.0 if "preferred_direction" in trace.decision.reasons else 0.5
        for trace in split_traces
    )
    background_boundary_alignment = _mean(
        max(
            trace.decision.split_candidate.color_strength,
            1.0
            if {"lsd", "layout_group", "component_boundary"}.intersection(
                trace.decision.split_candidate.sources
            )
            else 0.0,
        )
        for trace in split_traces
    )
    leaf_variances = [
        profile_by_bbox[leaf.bbox].features.color_variance
        if leaf.bbox in profile_by_bbox
        else _leaf_color_variance(image_pixels, leaf.bbox)
        for leaf in leaves
    ]
    visual_partition_score = 1.0 - _mean(leaf_variances)

    empty_region_penalty = sum(not content for content in leaf_contents) / max(1, len(leaves))
    thin_text_atomic_penalty = sum(_is_thin_text_leaf(leaf, content, split_tree.bbox) for leaf, content in zip(leaves, leaf_contents)) / max(1, len(leaves))
    single_small_element_penalty = sum(_is_small_single_element(content, split_tree.bbox) for content in leaf_contents) / max(1, len(leaves))
    broken_large_component_penalty = len(
        {
            element_id
            for trace in split_traces
            for element_id in trace.decision.split_candidate.large_component_crossing_ids
        }
    ) / max(1, len(elements))
    under_segmentation_penalty = _mean(
        _leaf_under_segmentation(leaf, content, decision_traces)
        for leaf, content in zip(leaves, leaf_contents)
    )
    generation_cost_penalty = min(1.0, len([content for content in leaf_contents if content]) / max(1, len(elements)))

    positive = {
        "separator_evidence": _clamp01(separator_evidence),
        "semantic_cohesion": _clamp01(semantic_cohesion),
        "layout_group_preservation": _clamp01(layout_group_preservation),
        "repeated_unit_preservation": _clamp01(repeated_unit_preservation),
        "direction_consistency": _clamp01(direction_consistency),
        "background_boundary_alignment": _clamp01(background_boundary_alignment),
        "visual_partition_score": _clamp01(visual_partition_score),
    }
    negative = {
        "empty_region_penalty": _clamp01(empty_region_penalty),
        "thin_text_atomic_penalty": _clamp01(thin_text_atomic_penalty),
        "single_small_element_penalty": _clamp01(single_small_element_penalty),
        "broken_large_component_penalty": _clamp01(broken_large_component_penalty),
        "under_segmentation_penalty": _clamp01(under_segmentation_penalty),
        "generation_cost_penalty": _clamp01(generation_cost_penalty),
    }
    total = sum(positive[name] * float(weights.get(name, 0.0)) for name in positive)
    total -= sum(negative[name] * float(weights.get(name, 0.0)) for name in negative)
    return TreeScoreBreakdown(round(total, 6), positive, negative)


def _leaf_nodes(root: SplitNode) -> list[SplitNode]:
    if not root.children:
        return [root]
    return [leaf for child in root.children for leaf in _leaf_nodes(child)]


def _profile_cohesion(profile: RegionProfile, decision: RegionDecisionCandidate) -> float:
    features = profile.features
    base = features.background_continuity * 0.35 + features.element_density * 0.2
    if profile.region_type in {"hero", "navigation", "text_section", "footer"}:
        base += features.text_ratio * 0.2 + features.largest_component_area_ratio * 0.25
    elif profile.region_type in {"card_grid", "list"}:
        base += (features.repeated_size_score + features.repeated_gap_score) * 0.225
    else:
        base += 0.2
    if decision.decision_type == "split" and "text_fragment" in decision.reasons:
        base -= 0.4
    return _clamp01(base)


def _leaf_color_variance(image_pixels: np.ndarray, bbox: BBox) -> float:
    left, top, right, bottom = bbox
    pixels = image_pixels[top:bottom, left:right]
    if pixels.size == 0:
        return 1.0
    return _clamp01(float(np.mean(np.std(pixels, axis=(0, 1)))) / 127.5)


def _is_thin_text_leaf(leaf: SplitNode, content: list[ElementNode], page_bbox: BBox) -> bool:
    page_height = max(1, page_bbox[3] - page_bbox[1])
    return bool(content) and all(element.is_text for element in content) and leaf.bbox[3] - leaf.bbox[1] <= max(96, page_height * 0.035)


def _is_small_single_element(content: list[ElementNode], page_bbox: BBox) -> bool:
    if len(content) != 1:
        return False
    element = content[0]
    page_area = max(1, (page_bbox[2] - page_bbox[0]) * (page_bbox[3] - page_bbox[1]))
    if element.area / page_area >= 0.04:
        return False
    if not element.is_text and (element.width >= 48 or element.height >= 48):
        return False
    return True


def _leaf_under_segmentation(leaf: SplitNode, content: list[ElementNode], decision_traces: tuple) -> float:
    if len(content) < 4:
        return 0.0
    trace = next((trace for trace in decision_traces if trace.profile.features.bbox == leaf.bbox), None)
    if trace is None:
        return min(1.0, len(content) / 12)
    features = trace.profile.features
    partition = max(
        features.strongest_horizontal_gap,
        features.strongest_vertical_gap,
        features.strongest_horizontal_separator,
        features.strongest_vertical_separator,
        features.horizontal_color_transition,
        features.vertical_color_transition,
    )
    return _clamp01(partition * min(1.0, len(content) / 8))


def _mean(values) -> float:
    materialized = list(values)
    return sum(materialized) / max(1, len(materialized))


def _clamp01(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)
