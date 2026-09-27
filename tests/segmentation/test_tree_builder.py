import unittest

from segmentation.layoutcoder_mvp import ElementBox, ProjectionTreeBuilder


class TreeBuilderTest(unittest.TestCase):
    def box(self, element_id, bbox):
        return ElementBox(element_id, bbox, "non_text")

    def builder(self, **overrides):
        settings = dict(max_depth=2, max_leaves=20, min_gap_px=8, min_child_width_px=10, min_child_height_px=10)
        settings.update(overrides)
        return ProjectionTreeBuilder(**settings)

    def test_two_columns_make_row(self):
        result = self.builder().build((100, 100), [self.box(0, (0, 0, 40, 100)), self.box(1, (60, 0, 100, 100))])
        self.assertEqual(result.root.node_type, "row")
        self.assertEqual([child.bbox for child in result.root.children], [(0, 0, 50, 100), (50, 0, 100, 100)])

    def test_two_rows_make_column(self):
        result = self.builder().build((100, 100), [self.box(0, (0, 0, 100, 40)), self.box(1, (0, 60, 100, 100))])
        self.assertEqual(result.root.node_type, "column")

    def test_grid_uses_stable_y_tie_break(self):
        boxes = [
            self.box(0, (0, 0, 40, 40)), self.box(1, (60, 0, 100, 40)),
            self.box(2, (0, 60, 40, 100)), self.box(3, (60, 60, 100, 100)),
        ]
        result = self.builder().build((100, 100), boxes)
        self.assertEqual(result.root.node_type, "column")
        self.assertTrue(all(child.node_type == "row" for child in result.root.children))

    def test_no_gap_is_atomic(self):
        result = self.builder().build((100, 100), [self.box(0, (0, 0, 100, 100))])
        self.assertEqual(result.root.node_type, "atomic")

    def test_max_depth(self):
        boxes = [self.box(0, (0, 0, 40, 40)), self.box(1, (60, 0, 100, 40)), self.box(2, (0, 60, 40, 100)), self.box(3, (60, 60, 100, 100))]
        result = self.builder(max_depth=1).build((100, 100), boxes)
        self.assertTrue(all(child.node_type == "atomic" for child in result.root.children))
        self.assertTrue(result.reached_max_depth)

    def test_max_leaves(self):
        boxes = [self.box(0, (0, 0, 40, 40)), self.box(1, (60, 0, 100, 40)), self.box(2, (0, 60, 40, 100)), self.box(3, (60, 60, 100, 100))]
        result = self.builder(max_leaves=2).build((100, 100), boxes)
        self.assertEqual(len(result.root.children), 2)
        self.assertTrue(result.reached_max_leaves)

    def test_split_line_never_crosses_box(self):
        boxes = [self.box(0, (0, 0, 45, 100)), self.box(1, (55, 0, 100, 100)), self.box(2, (48, 0, 52, 100))]
        result = self.builder(min_gap_px=2).build((100, 100), boxes)
        for entry in result.trace:
            if entry.outcome != "selected":
                continue
            position = entry.selected_gap.center
            for box in boxes:
                if entry.selected_gap.axis == "x":
                    self.assertFalse(box.bbox[0] < position < box.bbox[2])
                else:
                    self.assertFalse(box.bbox[1] < position < box.bbox[3])

    def test_deterministic(self):
        boxes = [self.box(0, (0, 0, 40, 100)), self.box(1, (60, 0, 100, 100))]
        first = self.builder().build((100, 100), boxes).root
        second = self.builder().build((100, 100), list(reversed(boxes))).root
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
