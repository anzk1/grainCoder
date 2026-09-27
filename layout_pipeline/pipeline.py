from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from segmentation.layoutcoder_mvp.uied_backend import UIEDBackend, UIEDDetectionArtifacts

from .atomic_merge import merge_weak_text_atomics
from .atomic_ownership import assign_atomic_ownership, replace_empty_atomics_with_spacers
from .layout_grouping import build_layout_groups
from .mask_layout_extractor import layout_from_split_tree
from .page_mask import PageMaskBuild, build_page_mask, select_page_mask_tree
from .schema import (
    BBox,
    ElementNode,
    ElementRole,
    LayoutGroup,
    LayoutNode,
    SeparatorLine,
    SplitCandidate,
    atomic_nodes,
    bbox_area,
)
from .separator_detection import detect_separator_lines
from .structure_validation import StructureValidation, validate_structure
from .tree_search import TreeSearchState, selected_split_candidates


@dataclass(frozen=True)
class LayoutAnalysis:
    artifacts: UIEDDetectionArtifacts
    elements: tuple[ElementNode, ...]
    layouts: tuple[LayoutGroup, ...]
    detected_separators: tuple[SeparatorLine, ...]
    page_mask: PageMaskBuild
    structure: LayoutNode
    validation: StructureValidation
    second_pass_used: bool
    chosen_tree_rank: int = 0
    rejected_trees: tuple[dict, ...] = ()


class LayoutCoderStructurePipeline:
    def __init__(self, config: dict, layoutcoder_root: str):
        reference = config.get("reference", {})
        self.backend = UIEDBackend(
            layoutcoder_root,
            expected_commit=reference.get("layoutcoder_commit"),
        )
        self.config = config

    def analyze(self, image_path: str, uied_output_dir: str | None = None) -> LayoutAnalysis:
        artifacts = self.backend.detect_artifacts(image_path, uied_output_dir)
        return self.analyze_artifacts(image_path, artifacts)

    def analyze_artifacts(self, image_path: str, artifacts: UIEDDetectionArtifacts) -> LayoutAnalysis:
        elements = _merged_elements(artifacts)
        detected_separators = detect_separator_lines(
            image_path,
            minimum_span_ratio=self.config.get("separator", {}).get("minimum_span_ratio", 0.5),
        )
        layouts = build_layout_groups(elements, detected_separators)
        mask_config = self.config.get("mask", {})
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
        page_mask = build_page_mask(
            source,
            elements,
            layouts,
            detected_separators,
            minimum_gap=mask_config.get("minimum_gap", 8),
            minimum_region_size=mask_config.get("minimum_region_size", 10),
            minimum_split_score=mask_config.get("minimum_split_score", 0.55),
            strong_separator_score=mask_config.get("strong_separator_score", 0.9),
            region_policies=self.config.get("region_policies", {}),
            tree_search_config=self.config.get("tree_search", {}),
            tree_score_config=self.config.get("tree_score", {}),
            large_component_area_ratio=mask_config.get("large_component_area_ratio", 0.07),
        )
        foreground_elements = [element for element in elements if element.segmentation_role == "foreground"]
        second_pass_used = False
        all_ranked_trees_are_atomic = bool(page_mask.ranked_states) and all(
            not state.split_tree.children for state in page_mask.ranked_states
        )
        fallback_tree_is_atomic = not page_mask.ranked_states and not page_mask.split_tree.children
        if (all_ranked_trees_are_atomic or fallback_tree_is_atomic) and len(foreground_elements) >= mask_config.get(
            "second_pass_min_foreground_elements", 8
        ):
            foreground_ids = {element.id for element in foreground_elements}
            foreground_layouts = [
                layout for layout in layouts if any(element_id in foreground_ids for element_id in layout.element_ids)
            ]
            page_mask = build_page_mask(
                source,
                foreground_elements,
                foreground_layouts,
                detected_separators,
                minimum_gap=mask_config.get("minimum_gap", 8),
                minimum_region_size=mask_config.get("minimum_region_size", 10),
                minimum_split_score=mask_config.get("minimum_split_score", 0.55),
                strong_separator_score=mask_config.get("strong_separator_score", 0.9),
                region_policies=self.config.get("region_policies", {}),
                tree_search_config=self.config.get("tree_search", {}),
                tree_score_config=self.config.get("tree_score", {}),
                large_component_area_ratio=mask_config.get("large_component_area_ratio", 0.07),
            )
            second_pass_used = True
        ranked_states = page_mask.ranked_states
        rejected_trees = []
        chosen_tree_rank = 0
        if ranked_states:
            selected_structure = None
            selected_validation = None
            selected_page_mask = page_mask
            first_failed_tree = None
            for rank, state in enumerate(ranked_states):
                candidate_page_mask = select_page_mask_tree(
                    page_mask,
                    state,
                    (artifacts.image_width, artifacts.image_height),
                )
                candidate_structure, candidate_validation = _build_validated_tree(
                    state,
                    candidate_page_mask.candidates,
                    elements,
                    layouts,
                    (artifacts.image_width, artifacts.image_height),
                    self.config.get("ownership", {}).get("icon_max_size", 48),
                    page_mask.strong_separator_score,
                    self.config.get("quality_gate", {}),
                )
                if rank == 0:
                    first_failed_tree = (candidate_structure, candidate_validation)
                if candidate_validation.passed:
                    selected_structure = candidate_structure
                    selected_validation = candidate_validation
                    selected_page_mask = candidate_page_mask
                    chosen_tree_rank = rank
                    rejected_trees.extend(
                        {
                            "rank": lower_rank,
                            "score": lower_state.global_score,
                            "tree": lower_state.split_tree.to_dict(),
                            "status": "not_selected_lower_rank",
                        }
                        for lower_rank, lower_state in enumerate(ranked_states[rank + 1 :], start=rank + 1)
                    )
                    break
                rejected_trees.append(
                    {
                        "rank": rank,
                        "score": state.global_score,
                        "tree": state.split_tree.to_dict(),
                        "validation": candidate_validation.to_dict(),
                    }
                )
            if selected_structure is None or selected_validation is None:
                state = ranked_states[0]
                selected_structure, selected_validation = first_failed_tree
                selected_page_mask = select_page_mask_tree(page_mask, state, source.size)
            structure = selected_structure
            validation = selected_validation
            page_mask = selected_page_mask
        else:
            structure = layout_from_split_tree(page_mask.split_tree)
            structure = assign_atomic_ownership(
                structure,
                elements,
                (artifacts.image_width, artifacts.image_height),
                icon_max_size=self.config.get("ownership", {}).get("icon_max_size", 48),
            )
            structure = replace_empty_atomics_with_spacers(structure)
            profiles = tuple(candidate_set.profile for candidate_set in page_mask.region_candidate_sets)
            structure = merge_weak_text_atomics(structure, elements, page_mask.strong_separator_score, profiles)
            validation = validate_structure(
                structure,
                elements,
                page_mask.candidates,
                page_mask.strong_separator_score,
                layouts,
                profiles,
                self.config.get("quality_gate", {}),
            )
        return LayoutAnalysis(
            artifacts,
            tuple(elements),
            tuple(layouts),
            tuple(detected_separators),
            page_mask,
            structure,
            validation,
            second_pass_used,
            chosen_tree_rank,
            tuple(rejected_trees),
        )


def _merged_elements(artifacts: UIEDDetectionArtifacts) -> list[ElementNode]:
    detected_height, detected_width = artifacts.merge_payload["img_shape"][:2]
    scale_x = artifacts.image_width / detected_width
    scale_y = artifacts.image_height / detected_height
    elements = []
    for index, component in enumerate(artifacts.merge_payload["compos"]):
        position = component.get("position", component)
        left = max(0, min(artifacts.image_width, round(position["column_min"] * scale_x)))
        top = max(0, min(artifacts.image_height, round(position["row_min"] * scale_y)))
        right = max(0, min(artifacts.image_width, round(position["column_max"] * scale_x)))
        bottom = max(0, min(artifacts.image_height, round(position["row_max"] * scale_y)))
        if right <= left or bottom <= top:
            continue
        elements.append(
            ElementNode(
                id=index,
                bbox=(left, top, right, bottom),
                uied_class=component.get("class", "Compo"),
                text=component.get("text_content"),
                segmentation_role=_segmentation_role(
                    component.get("class", "Compo"),
                    (left, top, right, bottom),
                    (artifacts.image_width, artifacts.image_height),
                    component.get("text_content"),
                ),
            )
        )
    return elements


def _build_validated_tree(
    state: TreeSearchState,
    split_candidates: tuple[SplitCandidate, ...],
    elements: list[ElementNode],
    layouts: list[LayoutGroup],
    page_size: tuple[int, int],
    icon_max_size: int,
    strong_separator_score: float,
    quality_gate: dict,
) -> tuple[LayoutNode, StructureValidation]:
    profiles = tuple(trace.profile for trace in state.decision_traces)
    selected_candidates = selected_split_candidates(state)
    available_candidate_ids = {
        (candidate.region, candidate.line.direction, candidate.line.position)
        for candidate in split_candidates
    }
    validation_candidates = split_candidates + tuple(
        candidate
        for candidate in selected_candidates
        if (candidate.region, candidate.line.direction, candidate.line.position) not in available_candidate_ids
    )
    structure = layout_from_split_tree(state.split_tree)
    structure = assign_atomic_ownership(structure, elements, page_size, icon_max_size=icon_max_size)
    structure = replace_empty_atomics_with_spacers(structure)
    structure = merge_weak_text_atomics(structure, elements, strong_separator_score, profiles)
    validation = validate_structure(
        structure,
        elements,
        validation_candidates,
        strong_separator_score,
        layouts,
        profiles,
        quality_gate,
    )
    return structure, validation


def _segmentation_role(
    uied_class: str,
    bbox: BBox,
    page_size: tuple[int, int],
    text: str | None,
) -> ElementRole:
    if uied_class == "Text" or text:
        return "foreground"
    page_width, page_height = page_size
    width_ratio = (bbox[2] - bbox[0]) / max(1, page_width)
    height_ratio = (bbox[3] - bbox[1]) / max(1, page_height)
    area_ratio = bbox_area(bbox) / max(1, page_width * page_height)
    if uied_class == "Block":
        if area_ratio >= 0.5 or (width_ratio >= 0.9 and height_ratio >= 0.4):
            return "background"
        return "container"
    if width_ratio >= 0.9 or area_ratio >= 0.15 or (width_ratio >= 0.65 and height_ratio >= 0.15):
        return "container"
    return "foreground"
