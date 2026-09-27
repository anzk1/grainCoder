import unittest
from types import SimpleNamespace

import numpy as np

from layout_pipeline.schema import ElementNode, SeparatorLine, SplitCandidate, SplitNode
from layout_pipeline.tree_scoring import score_split_tree


class TreeScoringTest(unittest.TestCase):
    def test_normalizes_score_terms_and_exempts_complete_large_component(self):
        tree = SplitNode((0, 0, 100, 100))
        elements = [ElementNode(0, (10, 10, 90, 90), "Compo")]

        breakdown = score_split_tree(
            tree,
            (),
            elements,
            [],
            np.zeros((100, 100, 3), dtype=np.float32),
            {"visual_partition_score": 0.1, "single_small_element_penalty": 0.2},
        )

        self.assertEqual(breakdown.negative["single_small_element_penalty"], 0.0)
        for value in breakdown.positive.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        for value in breakdown.negative.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_penalizes_tiny_single_text_leaf(self):
        breakdown = score_split_tree(
            SplitNode((0, 0, 100, 100)),
            (),
            [ElementNode(0, (10, 10, 20, 15), "Text", "line")],
            [],
            np.zeros((100, 100, 3), dtype=np.float32),
            {"single_small_element_penalty": 1.0},
        )

        self.assertEqual(breakdown.negative["single_small_element_penalty"], 1.0)

    def test_container_only_leaf_is_not_empty(self):
        tree = SplitNode(
            (0, 0, 100, 100),
            children=(SplitNode((0, 0, 50, 100)), SplitNode((50, 0, 100, 100))),
        )
        elements = [
            ElementNode(0, (10, 10, 40, 40), "Text", "copy"),
            ElementNode(1, (60, 10, 90, 90), "Block", segmentation_role="container"),
        ]

        breakdown = score_split_tree(
            tree,
            (),
            elements,
            [],
            np.zeros((100, 100, 3), dtype=np.float32),
            {"empty_region_penalty": 1.0},
        )

        self.assertEqual(breakdown.negative["empty_region_penalty"], 0.0)

    def test_component_boundary_counts_as_background_alignment(self):
        candidate = SplitCandidate(
            (0, 0, 100, 100),
            SeparatorLine("vertical", 50, 0, 100, "component_boundary"),
            ("component_boundary",),
            0.8,
            True,
            (),
            1,
            1,
            (),
            (),
            0,
            1.0,
        )
        features = SimpleNamespace(
            bbox=(0, 0, 100, 100),
            background_continuity=0.8,
            element_density=0.2,
            text_ratio=0.5,
            largest_component_area_ratio=0.2,
            repeated_size_score=0.0,
            repeated_gap_score=0.0,
            color_variance=0.1,
        )
        trace = SimpleNamespace(
            profile=SimpleNamespace(region_type="mixed", features=features),
            decision=SimpleNamespace(split_candidate=candidate, reasons=(), decision_type="split"),
        )
        tree = SplitNode(
            (0, 0, 100, 100),
            children=(SplitNode((0, 0, 50, 100)), SplitNode((50, 0, 100, 100))),
            split_line=candidate.line,
            split_score=candidate.score,
            split_sources=candidate.sources,
        )

        breakdown = score_split_tree(
            tree,
            (trace,),
            [ElementNode(0, (10, 10, 40, 40), "Text", "copy"), ElementNode(1, (60, 10, 90, 90), "Block")],
            [],
            np.zeros((100, 100, 3), dtype=np.float32),
            {"background_boundary_alignment": 1.0},
        )

        self.assertEqual(breakdown.positive["background_boundary_alignment"], 1.0)


if __name__ == "__main__":
    unittest.main()
