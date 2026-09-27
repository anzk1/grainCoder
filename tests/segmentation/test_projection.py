import unittest

from segmentation.layoutcoder_mvp import ElementBox, find_internal_gaps, find_projection_gaps, merge_intervals, project_boxes


class ProjectionTest(unittest.TestCase):
    def box(self, element_id, bbox):
        return ElementBox(element_id, bbox, "non_text")

    def test_merge_overlapping_and_touching(self):
        self.assertEqual(merge_intervals([(5, 10), (0, 5), (9, 12), (20, 25)]), [(0, 12), (20, 25)])

    def test_two_columns(self):
        boxes = [self.box(0, (0, 0, 40, 100)), self.box(1, (60, 0, 100, 100))]
        self.assertEqual(project_boxes(boxes, (0, 0, 100, 100), "x"), [(0, 40), (60, 100)])
        self.assertEqual(find_projection_gaps(boxes, (0, 0, 100, 100), 8)[0].axis, "x")

    def test_two_rows(self):
        boxes = [self.box(0, (0, 0, 100, 40)), self.box(1, (0, 60, 100, 100))]
        gap = find_projection_gaps(boxes, (0, 0, 100, 100), 8)[0]
        self.assertEqual((gap.axis, gap.start, gap.end), ("y", 40, 60))

    def test_edge_whitespace_is_not_gap(self):
        gaps = find_internal_gaps([(20, 80)], 0, 100, "x", 8)
        self.assertEqual(gaps, [])

    def test_equal_gap_prefers_y(self):
        boxes = [
            self.box(0, (0, 0, 40, 40)),
            self.box(1, (60, 0, 100, 40)),
            self.box(2, (0, 60, 40, 100)),
            self.box(3, (60, 60, 100, 100)),
        ]
        gaps = find_projection_gaps(boxes, (0, 0, 100, 100), 8)
        self.assertEqual([gap.axis for gap in gaps[:2]], ["y", "x"])


if __name__ == "__main__":
    unittest.main()
