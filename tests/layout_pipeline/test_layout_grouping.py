import unittest

from layout_pipeline.layout_grouping import build_layout_groups, is_semantic_layout_group
from layout_pipeline.schema import ElementNode, LayoutGroup, SeparatorLine


class LayoutGroupingTest(unittest.TestCase):
    def test_groups_text_and_non_text_separately(self):
        elements = [
            ElementNode(0, (10, 10, 40, 30), "Text"),
            ElementNode(1, (50, 10, 80, 30), "Text"),
            ElementNode(2, (10, 50, 40, 80), "Compo"),
            ElementNode(3, (50, 50, 80, 80), "Block"),
        ]

        groups = build_layout_groups(elements)

        self.assertEqual([group.uied_class for group in groups], ["Text", "Compo"])
        self.assertEqual([group.layout_type for group in groups], ["row", "row"])

    def test_separator_rejects_crossed_group(self):
        elements = [
            ElementNode(0, (10, 10, 40, 30), "Text"),
            ElementNode(1, (50, 10, 80, 30), "Text"),
        ]
        separator = SeparatorLine("vertical", 45, 0, 100, "lsd")

        self.assertEqual(build_layout_groups(elements, [separator]), [])

    def test_narrow_repeated_icon_column_is_not_an_indivisible_semantic_group(self):
        icon_column = LayoutGroup(0, "column", (100, 20, 120, 400), (1, 2, 3), "Compo")
        text_column = LayoutGroup(1, "column", (100, 20, 400, 400), (4, 5, 6), "Text")

        self.assertFalse(is_semantic_layout_group(icon_column))
        self.assertTrue(is_semantic_layout_group(text_column))


if __name__ == "__main__":
    unittest.main()
