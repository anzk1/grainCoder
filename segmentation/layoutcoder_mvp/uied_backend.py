from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .types import ElementBox
from .validation import prepare_element_boxes


class UIEDBackendError(RuntimeError):
    pass


LAYOUTCODER_UIED_PARAMETERS = {
    "min-grad": 10,
    "ffl-block": 5,
    "min-ele-area": 50,
    "merge-contained-ele": True,
    "merge-line-to-paragraph": True,
    "remove-bar": False,
    "max-line-gap": 30,
}


@dataclass(frozen=True)
class UIEDDetectionArtifacts:
    image_width: int
    image_height: int
    ip_payload: dict
    ocr_payload: dict
    merge_payload: dict

    def to_dict(self) -> dict:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "ip_payload": self.ip_payload,
            "ocr_payload": self.ocr_payload,
            "merge_payload": self.merge_payload,
        }


class UIEDBackend:
    def __init__(self, layoutcoder_root: str, timeout_seconds: int = 900, expected_commit: str | None = None):
        self.layoutcoder_root = Path(layoutcoder_root).resolve()
        self.timeout_seconds = timeout_seconds
        self.expected_commit = expected_commit
        if not (self.layoutcoder_root / "run_single.py").is_file():
            raise ValueError(f"LayoutCoder repository not found: {self.layoutcoder_root}")
        if expected_commit:
            completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.layoutcoder_root, text=True, capture_output=True, check=False)
            actual_commit = completed.stdout.strip()
            if completed.returncode != 0 or actual_commit != expected_commit:
                raise ValueError(f"LayoutCoder commit mismatch: expected {expected_commit}, found {actual_commit or "unknown"}")

    def detect(self, image_path: str, debug_dir: str | None = None) -> list[ElementBox]:
        artifacts = self.detect_artifacts(image_path, debug_dir)
        raw_payload = artifacts.merge_payload
        detected_shape = raw_payload["img_shape"]
        scale_x = artifacts.image_width / detected_shape[1]
        scale_y = artifacts.image_height / detected_shape[0]
        boxes = self._adapt_merged_elements(raw_payload, scale_x, scale_y)
        prepared = prepare_element_boxes(boxes, artifacts.image_width, artifacts.image_height)
        if debug_dir is not None:
            output_root = Path(debug_dir).resolve()
            (output_root / "uied_raw.json").write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
            (output_root / "uied_normalized.json").write_text(
                json.dumps([_element_box_payload(box) for box in prepared], indent=2), encoding="utf-8"
            )
        return prepared

    def detect_artifacts(self, image_path: str, debug_dir: str | None = None) -> UIEDDetectionArtifacts:
        source_image = Path(image_path).resolve()
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        output_owner = tempfile.TemporaryDirectory(prefix="dcgen-uied-") if debug_dir is None else None
        output_root = Path(output_owner.name if output_owner else debug_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-c",
            "from run_single import uied; import sys; uied(sys.argv[1], sys.argv[2])",
            str(source_image),
            str(output_root),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(self.layoutcoder_root), str(self.layoutcoder_root / "UIED")))
        try:
            completed = subprocess.run(
                command,
                cwd=self.layoutcoder_root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            if output_owner:
                output_owner.cleanup()
            raise UIEDBackendError(f"UIED timed out after {self.timeout_seconds} seconds") from exc
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            if output_owner:
                output_owner.cleanup()
            raise UIEDBackendError(f"UIED failed for {source_image.name}: {details}")
        try:
            return self.load_artifacts(str(source_image), str(output_root))
        finally:
            if output_owner:
                output_owner.cleanup()

    def load_artifacts(self, image_path: str, output_root: str) -> UIEDDetectionArtifacts:
        source_image = Path(image_path).resolve()
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        artifact_root = Path(output_root).resolve()
        sample_name = source_image.stem
        ip_payload = _load_uied_payload(artifact_root / "ip" / f"{sample_name}.json", "compos")
        ocr_payload = _load_uied_payload(artifact_root / "ocr" / f"{sample_name}.json", "texts")
        merge_payload = _load_uied_payload(artifact_root / "merge" / f"{sample_name}.json", "compos")
        with Image.open(source_image) as source:
            image_width, image_height = source.size
        return UIEDDetectionArtifacts(
            image_width=image_width,
            image_height=image_height,
            ip_payload=ip_payload,
            ocr_payload=ocr_payload,
            merge_payload=merge_payload,
        )

    @staticmethod
    def _adapt_merged_elements(raw_payload: dict, scale_x: float, scale_y: float) -> list[ElementBox]:
        boxes = []
        for index, component in enumerate(raw_payload.get("compos", [])):
            position = component.get("position", component)
            boxes.append(
                ElementBox(
                    element_id=index,
                    bbox=(
                        position["column_min"] * scale_x,
                        position["row_min"] * scale_y,
                        position["column_max"] * scale_x,
                        position["row_max"] * scale_y,
                    ),
                    kind="text" if component.get("class") == "Text" else "non_text",
                    source="uied",
                )
            )
        return boxes


def _load_uied_payload(path: Path, collection_key: str) -> dict:
    if not path.is_file():
        raise UIEDBackendError(f"UIED did not produce {collection_key} detections: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UIEDBackendError(f"UIED payload root must be a mapping: {path}")
    image_shape = payload.get("img_shape")
    if not isinstance(image_shape, list) or len(image_shape) < 2 or min(image_shape[:2]) <= 0:
        raise UIEDBackendError(f"UIED payload has invalid img_shape: {path}")
    if not isinstance(payload.get(collection_key), list):
        raise UIEDBackendError(f"UIED payload has invalid {collection_key}: {path}")
    return payload


def _element_box_payload(box: ElementBox) -> dict:
    return {
        "element_id": box.element_id,
        "bbox": list(box.bbox),
        "kind": box.kind,
        "confidence": box.confidence,
        "source": box.source,
    }
