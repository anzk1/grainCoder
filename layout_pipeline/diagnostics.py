from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from .pipeline import LayoutAnalysis
from .schema import LayoutNode


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class LayoutDiagnosticsWriter:
    def write(self, source_image_path: str, output_dir: str, analysis: LayoutAnalysis) -> None:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(output_root / "uied_raw.json", analysis.artifacts.to_dict())
        write_json(output_root / "merged_uied.json", {"elements": [element.to_dict() for element in analysis.elements]})
        write_json(output_root / "layouts.json", {"layouts": [layout.to_dict() for layout in analysis.layouts]})
        write_json(output_root / "detected_separators.json", {"separators": [line.to_dict() for line in analysis.detected_separators]})
        write_json(output_root / "separators.json", {"separators": [line.to_dict() for line in analysis.page_mask.separators]})
        write_json(
            output_root / "split_candidates.json",
            {"candidates": [candidate.to_dict() for candidate in analysis.page_mask.candidates]},
        )
        write_json(output_root / "split_tree.json", analysis.page_mask.split_tree.to_dict())
        write_json(output_root / "structure.json", analysis.structure.to_dict())
        write_json(output_root / "atomic_ownership.json", _ownership_payload(analysis.structure))
        write_json(output_root / "validation.json", analysis.validation.to_dict())
        write_json(
            output_root / "structure_parse.json",
            {"second_pass_used": analysis.second_pass_used, "chosen_tree_rank": analysis.chosen_tree_rank},
        )
        write_json(
            output_root / "region_profiles.json",
            {"profiles": [candidate_set.profile.to_dict() for candidate_set in analysis.page_mask.region_candidate_sets]},
        )
        write_json(
            output_root / "region_candidates.json",
            {"regions": [candidate_set.to_dict() for candidate_set in analysis.page_mask.region_candidate_sets]},
        )
        write_json(
            output_root / "beam_states.json",
            {
                "iterations": [
                    [state.to_dict() for state in iteration]
                    for iteration in analysis.page_mask.beam_history
                ],
                "ranked_complete_states": [state.to_dict() for state in analysis.page_mask.ranked_states],
            },
        )
        chosen_state = (
            analysis.page_mask.ranked_states[analysis.chosen_tree_rank]
            if analysis.page_mask.ranked_states
            else None
        )
        write_json(
            output_root / "tree_score_breakdown.json",
            chosen_state.score_breakdown.to_dict() if chosen_state is not None else {},
        )
        write_json(
            output_root / "chosen_tree.json",
            {
                "rank": analysis.chosen_tree_rank,
                "tree": analysis.page_mask.split_tree.to_dict(),
                "score": chosen_state.global_score if chosen_state is not None else None,
            },
        )
        write_json(output_root / "rejected_trees.json", {"trees": list(analysis.rejected_trees)})
        write_json(
            output_root / "quality_gate.json",
            {
                "passed": analysis.validation.passed,
                "chosen_tree_rank": analysis.chosen_tree_rank,
                "validation": analysis.validation.to_dict(),
            },
        )
        analysis.page_mask.mask.save(output_root / "page_mask.png")

        with Image.open(source_image_path) as opened:
            source = opened.convert("RGB")
        self._element_overlay(source, analysis, output_root / "uied_overlay.png")
        self._structure_overlay(source, analysis.structure, output_root / "structure_overlay.png")
        self._region_profile_overlay(source, analysis, output_root / "region_profile_overlay.png")

    @staticmethod
    def _element_overlay(source: Image.Image, analysis: LayoutAnalysis, output_path: Path) -> None:
        canvas = source.copy()
        draw = ImageDraw.Draw(canvas)
        colors = {"Text": "#ff2d55", "Compo": "#007aff", "Block": "#af52de"}
        for element in analysis.elements:
            color = colors.get(element.uied_class, "#34c759")
            draw.rectangle(element.bbox, outline=color, width=2)
            draw.text((element.bbox[0] + 2, element.bbox[1] + 2), str(element.id), fill=color)
        canvas.save(output_path)

    @staticmethod
    def _structure_overlay(source: Image.Image, root: LayoutNode, output_path: Path) -> None:
        canvas = source.copy()
        draw = ImageDraw.Draw(canvas)

        def visit(node: LayoutNode, depth: int) -> None:
            color = (37 * depth % 255, 97 * depth % 255, 173 * depth % 255)
            draw.rectangle(node.bbox, outline=color, width=max(1, 4 - min(depth, 3)))
            draw.text((node.bbox[0] + 2, node.bbox[1] + 2), f"{node.id}:{node.layout_type}", fill=color)
            for child in node.children:
                visit(child, depth + 1)

        visit(root, 1)
        canvas.save(output_path)

    @staticmethod
    def _region_profile_overlay(source: Image.Image, analysis: LayoutAnalysis, output_path: Path) -> None:
        canvas = source.copy()
        draw = ImageDraw.Draw(canvas)
        for candidate_set in analysis.page_mask.region_candidate_sets:
            profile = candidate_set.profile
            draw.rectangle(profile.features.bbox, outline="#ff9500", width=2)
            draw.text(
                (profile.features.bbox[0] + 2, profile.features.bbox[1] + 2),
                f"{profile.region_type}:{profile.confidence:.2f}",
                fill="#ff9500",
            )
        canvas.save(output_path)


def _ownership_payload(root: LayoutNode) -> dict:
    atomics = []

    def visit(node: LayoutNode) -> None:
        if node.layout_type == "atomic":
            atomics.append(
                {
                    "atomic_id": node.id,
                    "bbox": list(node.bbox),
                    "element_ids": list(node.element_ids),
                    "ownership_label": node.ownership_label,
                }
            )
        for child in node.children:
            visit(child)

    visit(root)
    return {"atomics": atomics}
