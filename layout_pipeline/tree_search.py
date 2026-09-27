from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .page_mask import RegionCandidateSet, _child_regions, build_region_candidate_set
from .schema import BBox, ElementNode, LayoutGroup, RegionDecisionCandidate, RegionProfile, SeparatorLine, SplitNode
from .tree_scoring import TreeScoreBreakdown, score_split_tree


@dataclass(frozen=True)
class PendingRegion:
    bbox: BBox
    depth: int

    def to_dict(self) -> dict:
        return {"bbox": list(self.bbox), "depth": self.depth}


@dataclass(frozen=True)
class RegionDecisionTrace:
    profile: RegionProfile
    decision: RegionDecisionCandidate
    depth: int

    def to_dict(self) -> dict:
        return {
            "profile": self.profile.to_dict(),
            "decision": self.decision.to_dict(),
            "depth": self.depth,
        }


@dataclass(frozen=True)
class TreeSearchState:
    split_tree: SplitNode
    pending_regions: tuple[PendingRegion, ...]
    local_score_sum: float
    global_score: float
    expansion_count: int
    decision_traces: tuple[RegionDecisionTrace, ...]
    score_breakdown: TreeScoreBreakdown

    def to_dict(self) -> dict:
        return {
            "split_tree": self.split_tree.to_dict(),
            "pending_regions": [region.to_dict() for region in self.pending_regions],
            "local_score_sum": self.local_score_sum,
            "global_score": self.global_score,
            "expansion_count": self.expansion_count,
            "decision_traces": [trace.to_dict() for trace in self.decision_traces],
            "score_breakdown": self.score_breakdown.to_dict(),
        }


@dataclass(frozen=True)
class TreeSearchOutcome:
    ranked_states: tuple[TreeSearchState, ...]
    region_candidate_sets: tuple[RegionCandidateSet, ...]
    beam_history: tuple[tuple[TreeSearchState, ...], ...]


def search_layout_trees(
    image_pixels: np.ndarray,
    page_size: tuple[int, int],
    elements: list[ElementNode],
    layouts: list[LayoutGroup],
    detected_lines: list[SeparatorLine],
    region_policies: dict,
    search_config: dict,
    score_config: dict,
    minimum_gap: int,
    minimum_region_size: int,
    minimum_split_score: float,
    large_component_area_ratio: float,
) -> TreeSearchOutcome:
    page_bbox = (0, 0, page_size[0], page_size[1])
    empty_breakdown = score_split_tree(SplitNode(page_bbox), (), elements, layouts, image_pixels, score_config)
    active_states = [TreeSearchState(SplitNode(page_bbox), (PendingRegion(page_bbox, 0),), 0.0, empty_breakdown.total_score, 0, (), empty_breakdown)]
    completed_states: list[TreeSearchState] = []
    candidate_cache: dict[BBox, RegionCandidateSet] = {}
    beam_history = [tuple(active_states)]
    beam_width = max(1, int(search_config.get("beam_width", 6)))
    max_split_candidates = max(1, int(search_config.get("max_split_candidates_per_region", 3)))
    max_expansions = max(1, int(search_config.get("max_expansions", 64)))
    max_depth = max(1, int(search_config.get("max_depth", 7)))
    partial_local_score_weight = float(search_config.get("partial_local_score_weight", 0.0))

    while active_states:
        expanded_states: list[TreeSearchState] = []
        for state in active_states:
            if not state.pending_regions:
                completed_states.append(state)
                continue
            if state.expansion_count >= max_expansions:
                completed_states.append(
                    _finalize_pending_regions(
                        state,
                        candidate_cache,
                        image_pixels,
                        page_size,
                        elements,
                        layouts,
                        detected_lines,
                        region_policies,
                        score_config,
                        minimum_gap,
                        minimum_region_size,
                        minimum_split_score,
                        large_component_area_ratio,
                        "max_expansions_guard",
                    )
                )
                continue

            pending = state.pending_regions[0]
            candidate_set = _candidate_set(
                candidate_cache,
                pending.bbox,
                image_pixels,
                page_size,
                elements,
                layouts,
                detected_lines,
                region_policies,
                minimum_gap,
                minimum_region_size,
                minimum_split_score,
                large_component_area_ratio,
            )
            decisions = list(candidate_set.decisions)
            if pending.depth >= max_depth:
                no_split = next(decision for decision in decisions if decision.decision_type == "no_split")
                decisions = [replace(no_split, reasons=no_split.reasons + ("max_depth_guard",))]
            else:
                no_split = [decision for decision in decisions if decision.decision_type == "no_split"]
                splits = [
                    decision
                    for decision in decisions
                    if decision.decision_type == "split"
                    and not {"raw_candidate_rejected", "below_region_minimum", "large_component_cut"}.intersection(decision.reasons)
                ][:max_split_candidates]
                decisions = no_split + splits

            for decision in decisions:
                expanded_states.append(
                    _expand_state(
                        state,
                        pending,
                        candidate_set.profile,
                        decision,
                        elements,
                        layouts,
                        image_pixels,
                        score_config,
                    )
                )
        if not expanded_states:
            break
        deduplicated = {state.split_tree.to_dict().__repr__(): state for state in expanded_states}
        ranked = sorted(
            deduplicated.values(),
            key=lambda state: _active_state_rank(state, partial_local_score_weight),
            reverse=True,
        )
        active_states = ranked[:beam_width]
        beam_history.append(tuple(active_states))

    completed_states.extend(state for state in active_states if not state.pending_regions)
    if not completed_states:
        completed_states = [
            _finalize_pending_regions(
                state,
                candidate_cache,
                image_pixels,
                page_size,
                elements,
                layouts,
                detected_lines,
                region_policies,
                score_config,
                minimum_gap,
                minimum_region_size,
                minimum_split_score,
                large_component_area_ratio,
                "search_exhausted",
            )
            for state in active_states
        ]
    ranked_completed = tuple(sorted(completed_states, key=_state_rank, reverse=True)[:beam_width])
    beam_history.append(ranked_completed)
    return TreeSearchOutcome(ranked_completed, tuple(candidate_cache.values()), tuple(beam_history))


def selected_split_candidates(state: TreeSearchState):
    return tuple(
        replace(trace.decision.split_candidate, selected=True)
        for trace in state.decision_traces
        if trace.decision.split_candidate is not None
    )


def _candidate_set(
    cache: dict[BBox, RegionCandidateSet],
    bbox: BBox,
    image_pixels: np.ndarray,
    page_size: tuple[int, int],
    elements: list[ElementNode],
    layouts: list[LayoutGroup],
    detected_lines: list[SeparatorLine],
    region_policies: dict,
    minimum_gap: int,
    minimum_region_size: int,
    minimum_split_score: float,
    large_component_area_ratio: float,
) -> RegionCandidateSet:
    if bbox not in cache:
        cache[bbox] = build_region_candidate_set(
            bbox,
            elements,
            layouts,
            detected_lines,
            image_pixels,
            page_size,
            minimum_gap,
            minimum_region_size,
            minimum_split_score,
            region_policies,
            large_component_area_ratio,
        )
    return cache[bbox]


def _expand_state(
    state: TreeSearchState,
    pending: PendingRegion,
    profile: RegionProfile,
    decision: RegionDecisionCandidate,
    elements: list[ElementNode],
    layouts: list[LayoutGroup],
    image_pixels: np.ndarray,
    score_config: dict,
) -> TreeSearchState:
    remaining = state.pending_regions[1:]
    if decision.split_candidate is None:
        replacement = SplitNode(pending.bbox, region_type=profile.region_type)
        pending_regions = remaining
    else:
        candidate = decision.split_candidate
        child_bboxes = _child_regions(pending.bbox, candidate.line)
        replacement = SplitNode(
            pending.bbox,
            tuple(SplitNode(bbox) for bbox in child_bboxes),
            candidate.line,
            decision.final_local_score,
            candidate.sources,
            profile.region_type,
        )
        pending_regions = tuple(PendingRegion(bbox, pending.depth + 1) for bbox in child_bboxes) + remaining
    split_tree = _replace_region(state.split_tree, pending.bbox, replacement)
    traces = state.decision_traces + (RegionDecisionTrace(profile, decision, pending.depth),)
    breakdown = score_split_tree(split_tree, traces, elements, layouts, image_pixels, score_config)
    return TreeSearchState(
        split_tree,
        pending_regions,
        round(state.local_score_sum + decision.final_local_score, 6),
        breakdown.total_score,
        state.expansion_count + 1,
        traces,
        breakdown,
    )


def _replace_region(root: SplitNode, bbox: BBox, replacement: SplitNode) -> SplitNode:
    if root.bbox == bbox and not root.children:
        return replacement
    return replace(root, children=tuple(_replace_region(child, bbox, replacement) for child in root.children))


def _finalize_pending_regions(
    state: TreeSearchState,
    candidate_cache: dict[BBox, RegionCandidateSet],
    image_pixels: np.ndarray,
    page_size: tuple[int, int],
    elements: list[ElementNode],
    layouts: list[LayoutGroup],
    detected_lines: list[SeparatorLine],
    region_policies: dict,
    score_config: dict,
    minimum_gap: int,
    minimum_region_size: int,
    minimum_split_score: float,
    large_component_area_ratio: float,
    reason: str,
) -> TreeSearchState:
    finalized = state
    for pending in state.pending_regions:
        candidate_set = _candidate_set(
            candidate_cache,
            pending.bbox,
            image_pixels,
            page_size,
            elements,
            layouts,
            detected_lines,
            region_policies,
            minimum_gap,
            minimum_region_size,
            minimum_split_score,
            large_component_area_ratio,
        )
        no_split = next(decision for decision in candidate_set.decisions if decision.decision_type == "no_split")
        finalized = _expand_state(
            replace(finalized, pending_regions=(pending,) + tuple(region for region in finalized.pending_regions if region != pending)),
            pending,
            candidate_set.profile,
            replace(no_split, reasons=no_split.reasons + (reason,)),
            elements,
            layouts,
            image_pixels,
            score_config,
        )
    return replace(finalized, pending_regions=())


def _state_rank(state: TreeSearchState) -> tuple[float, float, int, str]:
    average_local = state.local_score_sum / max(1, len(state.decision_traces))
    return (state.global_score, average_local, -state.expansion_count, state.split_tree.to_dict().__repr__())


def _active_state_rank(state: TreeSearchState, local_score_weight: float) -> tuple[float, float, int, str]:
    average_local = state.local_score_sum / max(1, len(state.decision_traces))
    return (
        state.global_score + average_local * local_score_weight,
        state.global_score,
        -state.expansion_count,
        state.split_tree.to_dict().__repr__(),
    )
