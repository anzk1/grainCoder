import unittest
from unittest.mock import patch

import numpy as np

from layout_pipeline.separator_detection import detect_separator_lines


class _Detector:
    def detect(self, image):
        lines = np.array([[[0, 50, 100, 50]], [[20, 0, 20, 100]]], dtype=np.float32)
        return lines, None, None, None


class SeparatorDetectionTest(unittest.TestCase):
    @patch("layout_pipeline.separator_detection.cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8))
    @patch("layout_pipeline.separator_detection.cv2.createLineSegmentDetector", return_value=_Detector())
    def test_preserves_lines_for_scored_crossing_evaluation(self, detector, imread):
        lines = detect_separator_lines("sample.png")

        self.assertEqual({line.direction for line in lines}, {"horizontal", "vertical"})


if __name__ == "__main__":
    unittest.main()
