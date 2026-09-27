import unittest

import numpy as np

from layout_pipeline.region_features import extract_region_features
from layout_pipeline.schema import ElementNode, LayoutGroup, SeparatorLine


class RegionFeaturesTest(unittest.TestCase):
    def test_normalizes_region_and_visual_features(self):
        image = np.full((100, 200, 3), 240, dtype=np.uint8)
        image[:, 100:] = 20
        elements = [
            ElementNode(0, (10, 10, 80, 25), "Text", "Title"),
            ElementNode(1, (10, 40, 80, 55), "Text", "Body"),
            ElementNode(2, (120, 10, 190, 80), "Compo"),
            ElementNode(3, (0, 0, 200, 100), "Block", segmentation_role="background"),
        ]
        layouts = [LayoutGroup(0, "column", (5, 5, 85, 60), (0, 1), "Text")]
        lines = [SeparatorLine("vertical", 100, 0, 100, "lsd")]

        features = extract_region_features(image, (0, 0, 200, 100), (200, 100), elements, layouts, lines)

        self.assertEqual(features.foreground_count, 3)
        self.assertEqual(features.text_count, 2)
        self.assertEqual(features.column_group_count, 1)
        self.assertEqual(features.strongest_vertical_separator, 1.0)
        self.assertGreater(features.vertical_color_transition, 0.5)
        for name, value in features.__dict__.items():
            if isinstance(value, float):
                self.assertGreaterEqual(value, 0.0, name)
                self.assertLessEqual(value, 1.0, name)

    def test_reclassifies_page_position_for_recursive_region(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        features = extract_region_features(image, (0, 85, 100, 100), (100, 100), [], [], [])

        self.assertEqual(features.page_position, "footer")
        self.assertEqual(features.page_height_ratio, 0.15)


if __name__ == "__main__":
    unittest.main()
