import unittest

from layout_pipeline.atomic_merge import merge_weak_text_atomics
from layout_pipeline.schema import ElementNode, LayoutNode


class AtomicMergeTest(unittest.TestCase):
    def test_merges_same_ownership_text_atomics_across_weak_separator(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 100),
            "column",
            (
                LayoutNode(1, (0, 0, 100, 50), "atomic", element_ids=(0,), ownership_label="content"),
                LayoutNode(2, (0, 50, 100, 100), "atomic", element_ids=(1,), ownership_label="content"),
            ),
            split_score=0.7,
            split_sources=("whitespace",),
        )
        elements = [
            ElementNode(0, (10, 10, 90, 30), "Text", "First"),
            ElementNode(1, (10, 60, 90, 80), "Text", "Second"),
        ]

        merged = merge_weak_text_atomics(root, elements, strong_separator_score=0.9)

        self.assertEqual(merged.layout_type, "atomic")
        self.assertEqual(merged.element_ids, (0, 1))

    def test_preserves_text_atomics_across_strong_separator(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 100),
            "column",
            (
                LayoutNode(1, (0, 0, 100, 50), "atomic", element_ids=(0,), ownership_label="content"),
                LayoutNode(2, (0, 50, 100, 100), "atomic", element_ids=(1,), ownership_label="content"),
            ),
            split_score=1.1,
            split_sources=("lsd",),
        )
        elements = [
            ElementNode(0, (10, 10, 90, 30), "Text", "First"),
            ElementNode(1, (10, 60, 90, 80), "Text", "Second"),
        ]

        merged = merge_weak_text_atomics(root, elements, strong_separator_score=0.9)

        self.assertEqual(merged.layout_type, "column")

    def test_merges_contiguous_text_and_small_icon_run(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 120),
            "column",
            (
                LayoutNode(1, (0, 0, 100, 40), "atomic", element_ids=(0,), ownership_label="footer", region_type="footer"),
                LayoutNode(2, (0, 40, 100, 80), "atomic", element_ids=(1,), ownership_label="footer", region_type="footer"),
                LayoutNode(3, (0, 80, 100, 120), "atomic", element_ids=(2,), ownership_label="footer", region_type="footer"),
            ),
            split_score=0.6,
            split_sources=("whitespace",),
            region_type="footer",
        )
        elements = [
            ElementNode(0, (10, 10, 90, 30), "Text", "Links"),
            ElementNode(1, (10, 50, 26, 66), "Compo"),
            ElementNode(2, (10, 90, 90, 110), "Text", "Legal"),
        ]

        merged = merge_weak_text_atomics(root, elements, strong_separator_score=0.9)

        self.assertEqual(merged.layout_type, "atomic")
        self.assertEqual(merged.element_ids, (0, 1, 2))

    def test_does_not_merge_card_grid_siblings(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 100),
            "row",
            (
                LayoutNode(1, (0, 0, 50, 100), "atomic", element_ids=(0,), ownership_label="content", region_type="card_grid"),
                LayoutNode(2, (50, 0, 100, 100), "atomic", element_ids=(1,), ownership_label="content", region_type="card_grid"),
            ),
            split_score=0.6,
            split_sources=("whitespace",),
            region_type="card_grid",
        )
        elements = [ElementNode(0, (5, 5, 45, 95), "Text", "Card A"), ElementNode(1, (55, 5, 95, 95), "Text", "Card B")]

        merged = merge_weak_text_atomics(root, elements, strong_separator_score=0.9)

        self.assertEqual(merged.layout_type, "row")
