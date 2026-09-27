import unittest

from segmentation.layoutcoder_mvp import ElementBox, intersection_area, ioa, iob, iou, normalize_overlaps


class OverlapNormalizerTest(unittest.TestCase):
    def box(self, element_id, bbox, confidence=None, kind="non_text"):
        return ElementBox(element_id, bbox, kind, confidence)

    def test_metrics(self):
        first = self.box(0, (0, 0, 10, 10))
        second = self.box(1, (5, 0, 15, 10))
        self.assertEqual(intersection_area(first, second), 50)
        self.assertAlmostEqual(iou(first, second), 1 / 3)
        self.assertEqual(ioa(first, second), 0.5)
        self.assertEqual(iob(first, second), 0.5)

    def test_duplicate_prefers_confidence(self):
        normalized = normalize_overlaps(
            [self.box(0, (0, 0, 10, 10), 0.2), self.box(1, (0, 0, 10, 10), 0.9)], 20, 20
        )
        self.assertEqual(len(normalized.boxes), 1)
        self.assertEqual(normalized.boxes[0].confidence, 0.9)
        self.assertEqual(normalized.operations[0].operation_type, "duplicate_keep_preferred")

    def test_containment_keeps_larger(self):
        normalized = normalize_overlaps(
            [self.box(0, (0, 0, 20, 20)), self.box(1, (2, 2, 5, 5), kind="text")], 30, 30
        )
        self.assertEqual([box.bbox for box in normalized.boxes], [(0, 0, 20, 20)])

    def test_partial_intersection_merges_chain(self):
        normalized = normalize_overlaps(
            [
                self.box(0, (0, 0, 10, 10)),
                self.box(1, (8, 0, 18, 10)),
                self.box(2, (16, 0, 26, 10)),
            ],
            30,
            20,
        )
        self.assertEqual([box.bbox for box in normalized.boxes], [(0, 0, 26, 10)])
        self.assertEqual(len(normalized.operations), 2)

    def test_disjoint_boxes_remain(self):
        normalized = normalize_overlaps(
            [self.box(0, (0, 0, 5, 5)), self.box(1, (10, 10, 15, 15))], 20, 20
        )
        self.assertEqual(len(normalized.boxes), 2)

    def test_clips_and_drops_invalid_boxes(self):
        normalized = normalize_overlaps(
            [self.box(0, (-2, -3, 4, 4)), self.box(1, (10, 10, 10, 15))], 12, 12
        )
        self.assertEqual([box.bbox for box in normalized.boxes], [(0, 0, 4, 4)])

    def test_deterministic(self):
        boxes = [self.box(8, (20, 0, 30, 10)), self.box(4, (0, 0, 10, 10))]
        first = normalize_overlaps(boxes, 40, 20)
        second = normalize_overlaps(list(reversed(boxes)), 40, 20)
        self.assertEqual(first.boxes, second.boxes)


if __name__ == "__main__":
    unittest.main()
