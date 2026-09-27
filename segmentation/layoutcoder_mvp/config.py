from __future__ import annotations

from pathlib import Path

import yaml

from .pipeline import LayoutCoderSegmentationPipeline
from .tree_builder import ProjectionTreeBuilder
from .uied_backend import UIEDBackend


def load_config(path: str) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload


def pipeline_from_config(config: dict, layoutcoder_root: str) -> LayoutCoderSegmentationPipeline:
    segmenter = config["segmenter"]
    overlap = config["overlap"]
    return LayoutCoderSegmentationPipeline(
        UIEDBackend(layoutcoder_root, expected_commit=config.get("reference", {}).get("layoutcoder_commit")),
        ProjectionTreeBuilder(
            max_depth=segmenter["max_depth"],
            max_leaves=segmenter["max_leaves"],
            min_gap_px=segmenter["min_gap_px"],
            min_child_width_px=segmenter["min_child_width_px"],
            min_child_height_px=segmenter["min_child_height_px"],
        ),
        containment_threshold=overlap["containment_threshold"],
        duplicate_iou_threshold=overlap["duplicate_iou_threshold"],
    )
