import unittest

from layout_pipeline.mask_layout_extractor import layout_from_split_tree
from layout_pipeline.schema import SeparatorLine, SplitNode, atomic_nodes


class MaskLayoutExtractorTest(unittest.TestCase):
    def test_alternating_splits_emit_only_canonical_types(self):
        top = SplitNode(
            (0, 0, 120, 60),
            (SplitNode((0, 0, 60, 60)), SplitNode((60, 0, 120, 60))),
            SeparatorLine("vertical", 60, 0, 60, "whitespace"),
            0.8,
            ("whitespace",),
        )
        bottom = SplitNode(
            (0, 60, 120, 120),
            (SplitNode((0, 60, 60, 120)), SplitNode((60, 60, 120, 120))),
            SeparatorLine("vertical", 60, 60, 120, "whitespace"),
            0.8,
            ("whitespace",),
        )
        split_root = SplitNode(
            (0, 0, 120, 120),
            (top, bottom),
            SeparatorLine("horizontal", 60, 0, 120, "whitespace"),
            0.8,
            ("whitespace",),
        )

        root = layout_from_split_tree(split_root)

        self.assertEqual(root.layout_type, "column")
        self.assertEqual({node.layout_type for node in atomic_nodes(root)}, {"atomic"})
        self.assertEqual(len(atomic_nodes(root)), 4)

    def test_no_separation_line_stops_as_atomic(self):
        root = layout_from_split_tree(SplitNode((0, 0, 80, 40)))

        self.assertEqual(root.layout_type, "atomic")


if __name__ == "__main__":
    unittest.main()
