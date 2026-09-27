from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from PIL import Image

from .dcgen_adapter import DCGenSegmentationNode, adapt_layout_tree
from .overlap_normalizer import normalize_overlaps
from .tree_builder import ProjectionTreeBuilder
from .types import ElementBox, LayoutNode, NormalizationResult, TreeBuildResult
from .uied_backend import UIEDBackend
from .visualization import draw_boxes, draw_layout_tree, save_image


@dataclass(frozen=True)
class SegmentationRun:
    detected_boxes: list[ElementBox]
    normalization: NormalizationResult
    tree: TreeBuildResult
    dcgen_root: DCGenSegmentationNode
    elapsed_seconds: float


class LayoutCoderSegmentationPipeline:
    def __init__(
        self,
        uied_backend: UIEDBackend,
        tree_builder: ProjectionTreeBuilder,
        containment_threshold: float = 0.95,
        duplicate_iou_threshold: float = 0.90,
    ):
        self.uied_backend = uied_backend
        self.tree_builder = tree_builder
        self.containment_threshold = containment_threshold
        self.duplicate_iou_threshold = duplicate_iou_threshold

    def segment(self, image_path: str, debug_dir: str | None = None) -> SegmentationRun:
        started_at = perf_counter()
        image = Image.open(image_path).convert("RGB")
        output_dir = Path(debug_dir) if debug_dir else None
        detected = self.uied_backend.detect(image_path, str(output_dir / "uied") if output_dir else None)
        normalization = normalize_overlaps(
            detected,
            image.width,
            image.height,
            containment_threshold=self.containment_threshold,
            duplicate_iou_threshold=self.duplicate_iou_threshold,
        )
        tree = self.tree_builder.build(image.size, normalization.boxes)
        dcgen_root = adapt_layout_tree(image, tree.root)
        run = SegmentationRun(detected, normalization, tree, dcgen_root, perf_counter() - started_at)
        if output_dir:
            self._save_diagnostics(output_dir, image, run)
        return run

    def _save_diagnostics(self, output_dir: Path, image: Image.Image, run: SegmentationRun) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_image(draw_boxes(image, run.detected_boxes), output_dir / "uied_overlay.png")
        save_image(draw_boxes(image, run.normalization.boxes), output_dir / "normalized_overlay.png")
        save_image(draw_layout_tree(image, run.tree.root), output_dir / "tree_overlay.png")
        (output_dir / "normalization_trace.json").write_text(
            json.dumps(
                {
                    "input_count": run.normalization.input_count,
                    "output_count": len(run.normalization.boxes),
                    "operations": [asdict(operation) for operation in run.normalization.operations],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "split_trace.json").write_text(
            json.dumps([asdict(entry) for entry in run.tree.trace], indent=2), encoding="utf-8"
        )
        (output_dir / "layout_tree.json").write_text(
            json.dumps(_layout_node_payload(run.tree.root), indent=2), encoding="utf-8"
        )
        run.dcgen_root.to_json_tree(str(output_dir / "dcgen_tree.json"))
        (output_dir / "segmentation_stats.json").write_text(
            json.dumps(
                {
                    "uied_box_count": len(run.detected_boxes),
                    "normalized_box_count": len(run.normalization.boxes),
                    "leaf_count": _leaf_count(run.tree.root),
                    "tree_depth": _tree_depth(run.tree.root),
                    "reached_max_depth": run.tree.reached_max_depth,
                    "reached_max_leaves": run.tree.reached_max_leaves,
                    "elapsed_seconds": run.elapsed_seconds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _layout_node_payload(node: LayoutNode) -> dict:
    return {
        "bbox": list(node.bbox),
        "node_type": node.node_type,
        "children": [_layout_node_payload(child) for child in node.children],
        "split_axis": node.split_axis,
        "split_position": node.split_position,
        "selected_gap": asdict(node.selected_gap) if node.selected_gap else None,
        "depth": node.depth,
    }


def _leaf_count(node: LayoutNode) -> int:
    return 1 if not node.children else sum(_leaf_count(child) for child in node.children)


def _tree_depth(node: LayoutNode) -> int:
    return 1 if not node.children else 1 + max(_tree_depth(child) for child in node.children)
