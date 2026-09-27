import unittest

import numpy as np
from PIL import Image, ImageDraw

from layout_pipeline.page_mask import build_page_mask, build_region_candidate_set
from layout_pipeline.schema import ElementNode, SeparatorLine


class PageMaskTest(unittest.TestCase):
    def test_discovers_recursive_whitespace_splits(self):
        elements = [
            ElementNode(0, (10, 10, 40, 40), "Compo"),
            ElementNode(1, (70, 10, 100, 40), "Compo"),
            ElementNode(2, (10, 70, 40, 100), "Compo"),
            ElementNode(3, (70, 70, 100, 100), "Compo"),
        ]

        built = build_page_mask(Image.new("RGB", (120, 120), "white"), elements, [], [], minimum_gap=8)

        self.assertGreaterEqual(len(built.separators), 2)
        self.assertEqual(len(built.split_tree.children), 2)
        self.assertGreater(len(built.ranked_states), 1)
        for line in built.separators:
            point = (
                (line.start + line.end) // 2,
                line.position,
            ) if line.direction == "horizontal" else (
                line.position,
                (line.start + line.end) // 2,
            )
            self.assertEqual(built.mask.getpixel(point), (0, 0, 0))

    def test_background_container_does_not_veto_foreground_split(self):
        elements = [
            ElementNode(0, (10, 10, 30, 40), "Text"),
            ElementNode(1, (90, 10, 110, 40), "Text"),
            ElementNode(2, (0, 0, 120, 60), "Block", segmentation_role="background"),
        ]

        built = build_page_mask(
            Image.new("RGB", (120, 60), "white"),
            elements,
            [],
            [SeparatorLine("vertical", 60, 0, 60, "lsd")],
        )

        selected = next(candidate for candidate in built.candidates if candidate.selected)
        self.assertEqual(selected.line.direction, "vertical")
        self.assertEqual(selected.nonforeground_crossing_ids, (2,))
        self.assertEqual(selected.large_component_crossing_ids, ())
        self.assertIn("lsd", selected.sources)

    def test_weak_whitespace_does_not_fragment_vertical_text_run(self):
        elements = [
            ElementNode(index, (20, 20 + index * 80, 180, 50 + index * 80), "Text", f"Item {index}")
            for index in range(4)
        ]

        source = Image.new("RGB", (200, 360), "white")
        draw = ImageDraw.Draw(source)
        for position in (60, 140, 220):
            draw.line((0, position, 200, position), fill="#dddddd", width=1)

        built = build_page_mask(source, elements, [], [])

        self.assertFalse(built.split_tree.children)
        self.assertTrue(any("semantic_text_run" in candidate.rejection_reasons for candidate in built.candidates))

    def test_sustained_background_change_is_strong_color_evidence(self):
        source = Image.new("RGB", (120, 120), "white")
        ImageDraw.Draw(source).rectangle((0, 0, 119, 59), fill="#0066cc")
        elements = [
            ElementNode(0, (20, 20, 100, 40), "Text", "Top"),
            ElementNode(1, (20, 80, 100, 100), "Text", "Bottom"),
        ]

        built = build_page_mask(source, elements, [], [])

        selected = next(candidate for candidate in built.candidates if candidate.selected)
        self.assertIn("color_change", selected.sources)
        self.assertGreaterEqual(selected.color_strength, 0.75)

    def test_large_component_edges_become_semantic_candidates(self):
        elements = [
            ElementNode(0, (10, 10, 80, 40), "Text", "Left"),
            ElementNode(1, (10, 55, 80, 85), "Text", "More"),
            ElementNode(2, (100, 5, 160, 95), "Block", segmentation_role="container"),
        ]

        candidate_set = build_region_candidate_set(
            (0, 0, 200, 100),
            elements,
            [],
            [],
            np.zeros((100, 200, 3), dtype=np.float32),
            (200, 100),
            8,
            10,
            0.45,
            {"mixed": {"minimum_split_score": 0.45}},
            0.08,
        )

        edge_candidates = [
            candidate
            for candidate in candidate_set.split_candidates
            if candidate.line.direction == "vertical" and candidate.line.position in {100, 160}
        ]
        self.assertTrue(edge_candidates)
        self.assertTrue(all(not candidate.large_component_crossing_ids for candidate in edge_candidates))
        self.assertTrue(all("component_boundary" in candidate.sources for candidate in edge_candidates))
        self.assertTrue(any(candidate.accepted for candidate in edge_candidates))


if __name__ == "__main__":
    unittest.main()
