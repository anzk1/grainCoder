import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from segmentation.layoutcoder_mvp.uied_backend import (
    LAYOUTCODER_UIED_PARAMETERS,
    UIEDBackend,
    UIEDDetectionArtifacts,
)


class UIEDBackendTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.layoutcoder_root = self.root / "LayoutCoder"
        self.layoutcoder_root.mkdir()
        (self.layoutcoder_root / "run_single.py").write_text("# fixture\n", encoding="utf-8")
        self.image_path = self.root / "sample.png"
        Image.new("RGB", (200, 100), "white").save(self.image_path)
        self.output_root = self.root / "uied"
        for directory in ("ip", "ocr", "merge"):
            (self.output_root / directory).mkdir(parents=True)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_payloads(self):
        payloads = {
            "ip": {
                "img_shape": [100, 200, 3],
                "compos": [{"id": 1, "class": "Block", "column_min": 0, "row_min": 0, "column_max": 200, "row_max": 80}],
            },
            "ocr": {
                "img_shape": [100, 200, 3],
                "texts": [{"id": 2, "content": "Hello", "column_min": 10, "row_min": 10, "column_max": 50, "row_max": 25}],
            },
            "merge": {
                "img_shape": [100, 200, 3],
                "compos": [{"id": 0, "class": "Text", "text_content": "Hello", "position": {"column_min": 10, "row_min": 10, "column_max": 50, "row_max": 25}}],
            },
        }
        for directory, payload in payloads.items():
            (self.output_root / directory / "sample.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_load_artifacts_preserves_raw_payloads(self):
        self._write_payloads()
        backend = UIEDBackend(str(self.layoutcoder_root))

        artifacts = backend.load_artifacts(str(self.image_path), str(self.output_root))

        self.assertEqual(artifacts.image_width, 200)
        self.assertEqual(artifacts.image_height, 100)
        self.assertEqual(artifacts.ocr_payload["texts"][0]["content"], "Hello")
        self.assertEqual(artifacts.ip_payload["compos"][0]["class"], "Block")

    def test_layoutcoder_uied_parameters_match_reference_pipeline(self):
        self.assertEqual(
            LAYOUTCODER_UIED_PARAMETERS,
            {
                "min-grad": 10,
                "ffl-block": 5,
                "min-ele-area": 50,
                "merge-contained-ele": True,
                "merge-line-to-paragraph": True,
                "remove-bar": False,
                "max-line-gap": 30,
            },
        )

    def test_detect_keeps_legacy_text_non_text_adaptation(self):
        backend = UIEDBackend(str(self.layoutcoder_root))
        artifacts = UIEDDetectionArtifacts(
            image_width=200,
            image_height=100,
            ip_payload={"img_shape": [100, 200, 3], "compos": []},
            ocr_payload={"img_shape": [100, 200, 3], "texts": []},
            merge_payload={
                "img_shape": [100, 200, 3],
                "compos": [
                    {"class": "Text", "position": {"column_min": 10, "row_min": 10, "column_max": 50, "row_max": 25}},
                    {"class": "Block", "position": {"column_min": 0, "row_min": 0, "column_max": 200, "row_max": 80}},
                ],
            },
        )

        with patch.object(backend, "detect_artifacts", return_value=artifacts):
            boxes = backend.detect(str(self.image_path))

        self.assertEqual([box.kind for box in boxes], ["non_text", "text"])
        self.assertEqual([box.bbox for box in boxes], [(0, 0, 200, 80), (10, 10, 50, 25)])

    def test_load_artifacts_rejects_invalid_collection(self):
        self._write_payloads()
        invalid_path = self.output_root / "ocr" / "sample.json"
        invalid_path.write_text(json.dumps({"img_shape": [100, 200, 3], "texts": {}}), encoding="utf-8")
        backend = UIEDBackend(str(self.layoutcoder_root))

        with self.assertRaisesRegex(RuntimeError, "invalid texts"):
            backend.load_artifacts(str(self.image_path), str(self.output_root))


if __name__ == "__main__":
    unittest.main()
