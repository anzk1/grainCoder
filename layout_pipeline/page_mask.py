from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from .schema import (
    BBox,
    ElementNode,
    LayoutGroup,
    RegionDecisionCandidate,
    RegionProfile,
    SeparatorLine,
    SeparatorSource,
    SplitCandidate,
    SplitNode,
    intersection_area,
)
from .region_classifier import classify_region
from .region_features import extract_region_features
from .split_policy import build_region_decisions
from .layout_grouping import is_semantic_layout_group

if TYPE_CHECKING:
    from .tree_search import TreeSearchState


@dataclass(frozen=True)
class PageMaskBuild:
    mask: Image.Image
    separators: tuple[SeparatorLine, ...]
    split_tree: SplitNode
    candidates: tuple[SplitCandidate, ...]
    strong_separator_score: float
    ranked_states: tuple["TreeSearchState", ...] = ()
    region_candidate_sets: tuple["RegionCandidateSet", ...] = ()
    beam_history: tuple[tuple["TreeSearchState", ...], ...] = ()


@dataclass(frozen=True)
class RegionCandidateSet:
    profile: RegionProfile
    split_candidates: tuple[SplitCandidate, ...]
    decisions: tuple[RegionDecisionCandidate, ...]

    def to_dict(self) -> dict:
        return {
            "profile": self.profile.to_dict(),
            "split_candidates": [candidate.to_dict() for candidate in self.split_candidates],
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


@dataclass
class _CandidateEvidence:
    direction: str
    position: int
    sources: set[SeparatorSource]
    gap_width: int = 0
    color_strength: float = 0.0


_SOURCE_WEIGHTS = {
    "lsd": 0.72,
    "whitespace": 0.42,
    "layout_group": 0.22,
    "component_boundary": 0.55,
    "color_change": 0.48,
}
_SOURCE_PRIORITY = ("lsd", "color_change", "component_boundary", "whitespace", "layout_group")


def _bbox_intersects_region(bbox: BBox, region: BBox) -> bool:
    return intersection_area(bbox, region) > 0


def _line_crosses_bbox(line: SeparatorLine, bbox: BBox) -> bool:
    if line.direction == "vertical":
        return bbox[0] < line.position < bbox[2] and max(bbox[1], line.start) < min(bbox[3], line.end)
    return bbox[1] < line.position < bbox[3] and max(bbox[0], line.start) < min(bbox[2], line.end)


def _merged_intervals(boxes: list[BBox], direction: str, region: BBox) -> list[tuple[int, int]]:
    if direction == "vertical":
        intervals = sorted((max(region[0], box[0]), min(region[2], box[2])) for box in boxes)
    else:
        intervals = sorted((max(region[1], box[1]), min(region[3], box[3])) for box in boxes)
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        if not merged or merged[-1][1] < start:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _color_transition_positions(
    image_pixels: np.ndarray,
    region: BBox,
    direction: str,
    maximum_candidates: int = 8,
) -> list[tuple[int, float]]:
    left, top, right, bottom = region
    region_pixels = image_pixels[top:bottom, left:right]
    if min(region_pixels.shape[:2]) < 3:
        return []
    if direction == "horizontal":
        color_profile = region_pixels.mean(axis=1)
        offset = top
    else:
        color_profile = region_pixels.mean(axis=0)
        offset = left
    differences = np.linalg.norm(np.diff(color_profile, axis=0), axis=1)
    if len(differences) < 3:
        return []
    smoothed = np.convolve(differences, np.ones(5, dtype=np.float32) / 5, mode="same")
    threshold = max(6.0, float(np.percentile(smoothed, 92)))
    local_maxima = [
        index
        for index in range(1, len(smoothed) - 1)
        if smoothed[index] >= threshold
        and smoothed[index] >= smoothed[index - 1]
        and smoothed[index] >= smoothed[index + 1]
    ]
    strongest = sorted(local_maxima, key=lambda index: (-smoothed[index], index))[:maximum_candidates]
    comparison_window = max(6, min(40, len(color_profile) // 20))
    sustained_transitions = []
    for index in strongest:
        before = color_profile[max(0, index - comparison_window + 1) : index + 1]
        after = color_profile[index + 1 : min(len(color_profile), index + comparison_window + 1)]
        if not len(before) or not len(after):
            continue
        sustained_change = float(np.linalg.norm(before.mean(axis=0) - after.mean(axis=0)))
        color_strength = min(1.0, sustained_change / 80.0)
        if color_strength >= 0.08:
            sustained_transitions.append((offset + index + 1, color_strength))
    return sorted(sustained_transitions)


def _add_candidate_evidence(
    evidence: list[_CandidateEvidence],
    direction: str,
    position: int,
    source: SeparatorSource,
    gap_width: int = 0,
    color_strength: float = 0.0,
) -> None:
    matching = next(
        (
            candidate
            for candidate in evidence
            if candidate.direction == direction and abs(candidate.position - position) <= 4
        ),
        None,
    )
    if matching is None:
        evidence.append(_CandidateEvidence(direction, position, {source}, gap_width, color_strength))
        return
    matching.sources.add(source)
    matching.gap_width = max(matching.gap_width, gap_width)
    matching.color_strength = max(matching.color_strength, color_strength)


def _collect_candidate_evidence(
    region: BBox,
    foreground_boxes: list[BBox],
    large_component_boxes: list[BBox],
    layouts: list[LayoutGroup],
    detected_lines: list[SeparatorLine],
    image_pixels: np.ndarray,
    minimum_gap: int,
) -> list[_CandidateEvidence]:
    evidence: list[_CandidateEvidence] = []
    for detected_line in detected_lines:
        if detected_line.direction == "horizontal":
            inside = region[1] < detected_line.position < region[3]
            overlap = max(0, min(region[2], detected_line.end) - max(region[0], detected_line.start))
            sufficient_span = overlap >= (region[2] - region[0]) * 0.5
        else:
            inside = region[0] < detected_line.position < region[2]
            overlap = max(0, min(region[3], detected_line.end) - max(region[1], detected_line.start))
            sufficient_span = overlap >= (region[3] - region[1]) * 0.5
        if inside and sufficient_span:
            _add_candidate_evidence(evidence, detected_line.direction, detected_line.position, "lsd")

    for direction in ("horizontal", "vertical"):
        intervals = _merged_intervals(foreground_boxes, direction, region)
        for previous_interval, current_interval in zip(intervals, intervals[1:]):
            gap_width = current_interval[0] - previous_interval[1]
            if gap_width < minimum_gap:
                continue
            position = round((previous_interval[1] + current_interval[0]) / 2)
            _add_candidate_evidence(evidence, direction, position, "whitespace", gap_width)

    for layout in layouts:
        if not _bbox_intersects_region(layout.bbox, region):
            continue
        for direction, positions in (
            ("vertical", (layout.bbox[0], layout.bbox[2])),
            ("horizontal", (layout.bbox[1], layout.bbox[3])),
        ):
            for position in positions:
                _add_candidate_evidence(evidence, direction, position, "layout_group")

    for component_bbox in large_component_boxes:
        component_width = component_bbox[2] - component_bbox[0]
        component_height = component_bbox[3] - component_bbox[1]
        supported_axes = []
        if component_height >= (region[3] - region[1]) * 0.5:
            supported_axes.append(("vertical", (component_bbox[0], component_bbox[2])))
        if component_width >= (region[2] - region[0]) * 0.5:
            supported_axes.append(("horizontal", (component_bbox[1], component_bbox[3])))
        for direction, positions in supported_axes:
            for position in positions:
                _add_candidate_evidence(evidence, direction, position, "component_boundary")

    for direction in ("horizontal", "vertical"):
        for position, color_strength in _color_transition_positions(image_pixels, region, direction):
            _add_candidate_evidence(
                evidence,
                direction,
                position,
                "color_change",
                color_strength=color_strength,
            )
    return sorted(evidence, key=lambda candidate: (candidate.direction, candidate.position))


def _text_run_axis(elements: list[ElementNode], region: BBox, page_size: tuple[int, int]) -> str | None:
    text_elements = [element for element in elements if element.is_text]
    non_text_elements = [element for element in elements if not element.is_text]
    if len(text_elements) < 2 or any(element.width > 48 or element.height > 48 for element in non_text_elements):
        return None
    thin_height = max(96, round(page_size[1] * 0.035))
    if median(element.height for element in text_elements) > thin_height:
        return None
    centers_x = [element.center[0] for element in text_elements]
    centers_y = [element.center[1] for element in text_elements]
    horizontal_spread = max(centers_x) - min(centers_x)
    vertical_spread = max(centers_y) - min(centers_y)
    region_width = region[2] - region[0]
    region_height = region[3] - region[1]
    vertical_run = horizontal_spread <= region_width * 0.35 and vertical_spread >= median(
        element.height for element in text_elements
    ) * 2
    horizontal_run = vertical_spread <= region_height * 0.35 and horizontal_spread >= median(
        element.width for element in text_elements
    ) * 2
    if vertical_run and not horizontal_run:
        return "vertical"
    if horizontal_run and not vertical_run:
        return "horizontal"
    if vertical_run and horizontal_run:
        vertical_ratio = vertical_spread / max(1, region_height)
        horizontal_ratio = horizontal_spread / max(1, region_width)
        return "vertical" if vertical_ratio >= horizontal_ratio else "horizontal"
    return None


def _candidate_line(region: BBox, evidence: _CandidateEvidence) -> SeparatorLine:
    source = next(source for source in _SOURCE_PRIORITY if source in evidence.sources)
    if evidence.direction == "horizontal":
        return SeparatorLine("horizontal", evidence.position, region[0], region[2], source)
    return SeparatorLine("vertical", evidence.position, region[1], region[3], source)


def _score_candidate(
    region: BBox,
    evidence: _CandidateEvidence,
    foreground_elements: list[ElementNode],
    nonforeground_elements: list[ElementNode],
    layouts: list[LayoutGroup],
    minimum_region_size: int,
    minimum_split_score: float,
    text_run_axis: str | None,
    large_component_area_ratio: float,
) -> SplitCandidate:
    line = _candidate_line(region, evidence)
    foreground_before_elements = []
    foreground_after_elements = []
    for element in foreground_elements:
        center_coordinate = element.center[0] if line.direction == "vertical" else element.center[1]
        (foreground_before_elements if center_coordinate < line.position else foreground_after_elements).append(element)
    foreground_crossing = tuple(element.id for element in foreground_elements if _line_crosses_bbox(line, element.bbox))
    nonforeground_crossing = tuple(
        element.id for element in nonforeground_elements if _line_crosses_bbox(line, element.bbox)
    )
    layout_group_crossing = tuple(
        layout.group_id
        for layout in layouts
        if is_semantic_layout_group(layout) and _line_crosses_bbox(line, layout.bbox)
    )
    region_area = max(1, (region[2] - region[0]) * (region[3] - region[1]))
    partition_elements = foreground_elements + [
        element
        for element in nonforeground_elements
        if intersection_area(element.bbox, region) / region_area >= large_component_area_ratio
    ]
    content_before = []
    content_after = []
    for element in partition_elements:
        center_coordinate = element.center[0] if line.direction == "vertical" else element.center[1]
        (content_before if center_coordinate < line.position else content_after).append(element)
    component_boundary_supported = (
        "lsd" in evidence.sources
        or "layout_group" in evidence.sources
        or "component_boundary" in evidence.sources
        or evidence.color_strength >= 0.75
    )
    large_component_crossing = () if component_boundary_supported else tuple(
        element.id
        for element in foreground_elements + nonforeground_elements
        if _line_crosses_bbox(line, element.bbox)
        and intersection_area(element.bbox, region) / region_area >= large_component_area_ratio
    )
    rejection_reasons: list[str] = []
    region_start, region_end = (region[0], region[2]) if line.direction == "vertical" else (region[1], region[3])
    first_size = line.position - region_start
    second_size = region_end - line.position
    if first_size < minimum_region_size or second_size < minimum_region_size:
        rejection_reasons.append("child_too_small")
    if len(foreground_elements) < 2:
        rejection_reasons.append("insufficient_foreground")
    if not content_before or not content_after:
        rejection_reasons.append("empty_foreground_child")
    strong_semantic_evidence = (
        "lsd" in evidence.sources
        or "component_boundary" in evidence.sources
        or evidence.color_strength >= 0.75
    )
    if text_run_axis == "vertical" and line.direction == "horizontal" and not strong_semantic_evidence:
        rejection_reasons.append("semantic_text_run")
    if text_run_axis == "horizontal" and line.direction == "vertical" and not strong_semantic_evidence:
        rejection_reasons.append("semantic_text_run")
    if (
        not strong_semantic_evidence
        and any(
            layout.uied_class == "Text"
            and len(layout.element_ids) >= 2
            and _line_crosses_bbox(line, layout.bbox)
            for layout in layouts
        )
    ):
        rejection_reasons.append("semantic_layout_group")

    foreground_total = max(1, len(foreground_elements))
    child_balance = min(len(content_before), len(content_after)) / max(1, max(len(content_before), len(content_after)))
    area_balance = min(first_size, second_size) / max(1, max(first_size, second_size))
    source_score = sum(
        _SOURCE_WEIGHTS[source]
        for source in evidence.sources
        if source != "color_change"
    ) + _SOURCE_WEIGHTS["color_change"] * evidence.color_strength
    gap_score = min(0.28, evidence.gap_width / max(1, region_end - region_start) * 2)
    crossing_penalty = len(foreground_crossing) / foreground_total * 1.35 + len(foreground_crossing) * 0.08
    container_penalty = min(0.12, len(nonforeground_crossing) * 0.015)
    score = source_score + gap_score + child_balance * 0.34 + area_balance * 0.16
    score -= crossing_penalty + container_penalty
    if score < minimum_split_score:
        rejection_reasons.append("score_below_threshold")
    return SplitCandidate(
        region=region,
        line=line,
        sources=tuple(source for source in _SOURCE_PRIORITY if source in evidence.sources),
        score=round(score, 6),
        accepted=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
        foreground_before=len(foreground_before_elements),
        foreground_after=len(foreground_after_elements),
        foreground_crossing_ids=foreground_crossing,
        nonforeground_crossing_ids=nonforeground_crossing,
        gap_width=evidence.gap_width,
        child_balance=round(child_balance, 6),
        color_strength=round(evidence.color_strength, 6),
        layout_group_crossing_ids=layout_group_crossing,
        large_component_crossing_ids=large_component_crossing,
    )


def _child_regions(region: BBox, line: SeparatorLine) -> tuple[BBox, BBox]:
    if line.direction == "vertical":
        return (
            (region[0], region[1], line.position, region[3]),
            (line.position, region[1], region[2], region[3]),
        )
    return (
        (region[0], region[1], region[2], line.position),
        (region[0], line.position, region[2], region[3]),
    )


def build_region_candidate_set(
    region: BBox,
    elements: list[ElementNode],
    layouts: list[LayoutGroup],
    detected_lines: list[SeparatorLine],
    image_pixels: np.ndarray,
    page_size: tuple[int, int],
    minimum_gap: int,
    minimum_region_size: int,
    minimum_split_score: float,
    region_policies: dict,
    large_component_area_ratio: float,
) -> RegionCandidateSet:
    region_elements = [element for element in elements if _bbox_intersects_region(element.bbox, region)]
    foreground_elements = [element for element in region_elements if element.segmentation_role == "foreground"]
    nonforeground_elements = [element for element in region_elements if element.segmentation_role != "foreground"]
    relevant_layouts = [layout for layout in layouts if _bbox_intersects_region(layout.bbox, region)]
    region_area = max(1, (region[2] - region[0]) * (region[3] - region[1]))
    large_component_boxes = [
        element.bbox
        for element in region_elements
        if not element.is_text and intersection_area(element.bbox, region) / region_area >= large_component_area_ratio
    ]
    evidence = _collect_candidate_evidence(
        region,
        [element.bbox for element in foreground_elements],
        large_component_boxes,
        relevant_layouts,
        detected_lines,
        image_pixels,
        minimum_gap,
    )
    text_run_axis = _text_run_axis(foreground_elements, region, page_size)
    evaluated = [
        _score_candidate(
            region,
            candidate_evidence,
            foreground_elements,
            nonforeground_elements,
            relevant_layouts,
            minimum_region_size,
            minimum_split_score,
            text_run_axis,
            large_component_area_ratio,
        )
        for candidate_evidence in evidence
    ]
    profile = classify_region(
        extract_region_features(
            image_pixels,
            region,
            page_size,
            region_elements,
            relevant_layouts,
            detected_lines,
        )
    )
    policy = region_policies.get(profile.region_type, region_policies.get("mixed", {}))
    decisions = build_region_decisions(
        profile,
        evaluated,
        region_elements,
        relevant_layouts,
        policy,
    )
    return RegionCandidateSet(profile, tuple(evaluated), decisions)


def _render_page_mask(image_size: tuple[int, int], separators: list[SeparatorLine]) -> Image.Image:
    mask = Image.new("RGB", image_size, (147, 112, 219))
    draw = ImageDraw.Draw(mask)
    for line in separators:
        if line.direction == "horizontal":
            draw.line((line.start, line.position, line.end, line.position), fill="black", width=5)
        else:
            draw.line((line.position, line.start, line.position, line.end), fill="black", width=5)
    return mask


def build_page_mask(
    image: Image.Image,
    elements: list[ElementNode],
    layouts: list[LayoutGroup],
    detected_lines: list[SeparatorLine],
    minimum_gap: int = 8,
    minimum_region_size: int = 10,
    minimum_split_score: float = 0.55,
    strong_separator_score: float = 0.9,
    region_policies: dict | None = None,
    tree_search_config: dict | None = None,
    tree_score_config: dict | None = None,
    large_component_area_ratio: float = 0.07,
) -> PageMaskBuild:
    from .tree_search import search_layout_trees

    source = image.convert("RGB")
    image_pixels = np.asarray(source, dtype=np.float32)
    outcome = search_layout_trees(
        image_pixels,
        source.size,
        elements,
        layouts,
        detected_lines,
        region_policies or {"mixed": {}},
        tree_search_config or {},
        tree_score_config or {},
        minimum_gap=minimum_gap,
        minimum_region_size=minimum_region_size,
        minimum_split_score=minimum_split_score,
        large_component_area_ratio=large_component_area_ratio,
    )
    top_state = outcome.ranked_states[0]
    selected_split_candidates = {
        _candidate_identity(trace.decision.split_candidate)
        for trace in top_state.decision_traces
        if trace.decision.split_candidate is not None
    }
    candidate_trace = tuple(
        replace(candidate, selected=_candidate_identity(candidate) in selected_split_candidates)
        for candidate_set in outcome.region_candidate_sets
        for candidate in candidate_set.split_candidates
    )
    selected_lines = [
        trace.decision.split_candidate.line
        for trace in top_state.decision_traces
        if trace.decision.split_candidate is not None
    ]
    return PageMaskBuild(
        _render_page_mask(source.size, selected_lines),
        tuple(selected_lines),
        top_state.split_tree,
        candidate_trace,
        strong_separator_score,
        outcome.ranked_states,
        outcome.region_candidate_sets,
        outcome.beam_history,
    )


def select_page_mask_tree(page_mask: PageMaskBuild, state: "TreeSearchState", image_size: tuple[int, int]) -> PageMaskBuild:
    selected_split_candidates = {
        _candidate_identity(trace.decision.split_candidate)
        for trace in state.decision_traces
        if trace.decision.split_candidate is not None
    }
    candidates = tuple(
        replace(candidate, selected=_candidate_identity(candidate) in selected_split_candidates)
        for candidate in page_mask.candidates
    )
    separators = tuple(
        trace.decision.split_candidate.line
        for trace in state.decision_traces
        if trace.decision.split_candidate is not None
    )
    return replace(
        page_mask,
        mask=_render_page_mask(image_size, list(separators)),
        separators=separators,
        split_tree=state.split_tree,
        candidates=candidates,
    )


def _candidate_identity(candidate: SplitCandidate) -> tuple[BBox, str, int]:
    return candidate.region, candidate.line.direction, candidate.line.position
