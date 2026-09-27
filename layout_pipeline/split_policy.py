from __future__ import annotations

from .layout_grouping import is_semantic_layout_group
from .schema import ElementNode, LayoutGroup, RegionDecisionCandidate, RegionProfile, SplitCandidate


def build_region_decisions(
    profile: RegionProfile,
    split_candidates: list[SplitCandidate],
    elements: list[ElementNode],
    layouts: list[LayoutGroup],
    policy: dict,
) -> tuple[RegionDecisionCandidate, ...]:
    features = profile.features
    strongest_separator = max(
        features.strongest_horizontal_separator,
        features.strongest_vertical_separator,
        features.horizontal_color_transition,
        features.vertical_color_transition,
    )
    repetition = (features.repeated_size_score + features.repeated_gap_score) / 2
    semantic_completeness = _semantic_completeness(profile)
    disconnected_group_penalty = min(0.4, (features.row_group_count + features.column_group_count + features.grid_group_count) * 0.08)
    under_segmentation_penalty = min(0.45, strongest_separator * 0.25 + disconnected_group_penalty)
    no_split_score = (
        float(policy.get("no_split_bias", 0.0))
        + semantic_completeness * 0.35
        + features.background_continuity * 0.2
        + features.largest_component_area_ratio * 0.2
        - strongest_separator * 0.25
        - disconnected_group_penalty
        - under_segmentation_penalty
    )
    decisions = [
        RegionDecisionCandidate(
            "no_split",
            None,
            round(semantic_completeness, 6),
            round(no_split_score - semantic_completeness, 6),
            round(no_split_score, 6),
            (
                f"region_type:{profile.region_type}",
                f"background_continuity:{features.background_continuity:.3f}",
                f"strongest_separator:{strongest_separator:.3f}",
            ),
        )
    ]

    deduplicated = _deduplicate_candidates(split_candidates)
    evaluated_splits = [
        _split_decision(profile, candidate, elements, layouts, policy, repetition)
        for candidate in deduplicated
    ]
    axis_winners: list[RegionDecisionCandidate] = []
    for direction in ("horizontal", "vertical"):
        matching = [
            decision
            for decision in evaluated_splits
            if decision.split_candidate is not None and decision.split_candidate.line.direction == direction
        ]
        if matching:
            axis_winners.append(max(matching, key=lambda decision: decision.final_local_score))
    semantic_candidates = [
        decision
        for decision in evaluated_splits
        if decision.split_candidate is not None
        and {"layout_group", "component_boundary"}.intersection(decision.split_candidate.sources)
        and decision not in axis_winners
    ]
    if semantic_candidates:
        axis_winners.append(max(semantic_candidates, key=lambda decision: decision.final_local_score))

    decisions.extend(axis_winners[:3])
    return tuple(sorted(decisions, key=lambda decision: (-decision.final_local_score, decision.decision_type)))


def _semantic_completeness(profile: RegionProfile) -> float:
    features = profile.features
    if profile.region_type in {"hero", "navigation", "text_section", "footer"}:
        return min(1.0, features.background_continuity * 0.45 + features.text_ratio * 0.25 + features.largest_component_area_ratio * 0.3)
    if profile.region_type in {"card_grid", "list"}:
        return min(1.0, (features.repeated_size_score + features.repeated_gap_score) / 2)
    return min(1.0, features.element_density + features.background_continuity * 0.25)


def _deduplicate_candidates(split_candidates: list[SplitCandidate]) -> list[SplitCandidate]:
    kept: list[SplitCandidate] = []
    for candidate in sorted(split_candidates, key=_candidate_rank, reverse=True):
        if any(
            existing.line.direction == candidate.line.direction
            and abs(existing.line.position - candidate.line.position) < 8
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _candidate_rank(candidate: SplitCandidate) -> tuple[float, int, int, int]:
    return (candidate.score, len(candidate.sources), candidate.gap_width, -candidate.line.position)


def _split_decision(
    profile: RegionProfile,
    candidate: SplitCandidate,
    elements: list[ElementNode],
    layouts: list[LayoutGroup],
    policy: dict,
    repetition: float,
) -> RegionDecisionCandidate:
    reasons = list(candidate.rejection_reasons)
    adjustment = 0.0
    preferred_directions = tuple(policy.get("preferred_directions", ()))
    if candidate.line.direction in preferred_directions:
        adjustment += float(policy.get("preferred_direction_bonus", 0.0))
        reasons.append("preferred_direction")
    if {"layout_group", "component_boundary"}.intersection(candidate.sources):
        adjustment += float(policy.get("layout_group_bonus", 0.0))
        reasons.append("layout_group_boundary")
    if repetition >= 0.6 and profile.region_type in {"card_grid", "list"}:
        adjustment += float(policy.get("repeated_unit_bonus", 0.0)) * repetition
        reasons.append("repeated_unit_boundary")

    large_crossings = candidate.large_component_crossing_ids
    strong_boundary = (
        "lsd" in candidate.sources
        or candidate.color_strength >= 0.75
        or {"layout_group", "component_boundary"}.intersection(candidate.sources)
    )
    if large_crossings and not strong_boundary:
        adjustment -= float(policy.get("large_component_cut_penalty", 0.0))
        reasons.append("large_component_cut")

    broken_groups = [
        layout
        for layout in layouts
        if is_semantic_layout_group(layout) and _line_crosses_bbox(candidate, layout.bbox)
    ]
    if broken_groups:
        adjustment -= float(policy.get("broken_layout_group_penalty", 0.35))
        reasons.append("broken_layout_group")

    if profile.region_type in {"navigation", "text_section", "footer"} and candidate.foreground_crossing_ids:
        adjustment -= float(policy.get("text_fragment_penalty", 0.0))
        reasons.append("text_fragment")

    minimum_score = float(policy.get("minimum_split_score", 0.0))
    final_score = candidate.score + adjustment
    if final_score < minimum_score:
        reasons.append("below_region_minimum")
    if not candidate.accepted:
        reasons.append("raw_candidate_rejected")
    return RegionDecisionCandidate(
        "split",
        candidate,
        candidate.score,
        round(adjustment, 6),
        round(final_score, 6),
        tuple(dict.fromkeys(reasons)),
    )


def _line_crosses_bbox(candidate: SplitCandidate, bbox: tuple[int, int, int, int]) -> bool:
    line = candidate.line
    if line.direction == "vertical":
        return bbox[0] < line.position < bbox[2] and max(bbox[1], line.start) < min(bbox[3], line.end)
    return bbox[1] < line.position < bbox[3] and max(bbox[0], line.start) < min(bbox[2], line.end)
