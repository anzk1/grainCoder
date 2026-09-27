import unittest
from dataclasses import replace

from layout_pipeline.region_classifier import classify_region
from layout_pipeline.schema import RegionFeatures


def make_region_features(**changes) -> RegionFeatures:
    features = RegionFeatures(
        bbox=(0, 0, 100, 100),
        page_width_ratio=0.5,
        page_height_ratio=0.5,
        page_area_ratio=0.25,
        page_position="content",
        foreground_count=4,
        text_count=2,
        non_text_count=2,
        text_ratio=0.5,
        element_density=0.3,
        median_text_height=0.08,
        text_line_count=2,
        horizontal_text_spread=0.4,
        vertical_text_spread=0.4,
        largest_component_area_ratio=0.1,
        container_coverage_ratio=0.1,
        row_group_count=0,
        column_group_count=0,
        grid_group_count=0,
        repeated_size_score=0.1,
        repeated_gap_score=0.1,
        strongest_horizontal_gap=0.05,
        strongest_vertical_gap=0.05,
        strongest_horizontal_separator=0.0,
        strongest_vertical_separator=0.0,
        color_variance=0.1,
        horizontal_color_transition=0.1,
        vertical_color_transition=0.1,
        background_continuity=0.8,
        edge_density=0.1,
    )
    return replace(features, **changes)


class RegionClassifierTest(unittest.TestCase):
    def test_classifies_representative_region_types(self):
        cases = {
            "hero": make_region_features(page_area_ratio=0.8, largest_component_area_ratio=0.55, text_count=2),
            "navigation": make_region_features(page_position="header", page_height_ratio=0.08, horizontal_text_spread=0.9, text_ratio=0.9),
            "text_section": make_region_features(text_ratio=1.0, non_text_count=0, vertical_text_spread=0.8),
            "card_grid": make_region_features(foreground_count=10, row_group_count=2, grid_group_count=1, repeated_size_score=0.95, repeated_gap_score=0.9),
            "two_column": make_region_features(strongest_vertical_gap=0.8, strongest_horizontal_gap=0.0, column_group_count=2),
            "list": make_region_features(text_ratio=0.8, vertical_text_spread=0.9, row_group_count=2, repeated_size_score=0.8, repeated_gap_score=0.9),
            "footer": make_region_features(page_position="footer", foreground_count=9, text_ratio=0.8, column_group_count=2),
        }
        for expected, features in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(classify_region(features).region_type, expected)

    def test_uses_mixed_when_no_specialized_profile_is_strong(self):
        profile = classify_region(
            make_region_features(
                foreground_count=1,
                text_count=0,
                non_text_count=1,
                text_ratio=0.0,
                element_density=0.05,
                background_continuity=0.2,
            )
        )

        self.assertEqual(profile.region_type, "mixed")

    def test_two_column_profile_applies_to_content_band_not_whole_page_stack(self):
        full_page = make_region_features(
            page_height_ratio=1.0,
            strongest_vertical_gap=0.25,
            strongest_horizontal_gap=0.14,
            text_ratio=0.57,
            non_text_count=3,
            vertical_text_spread=0.7,
        )
        content_band = replace(full_page, page_height_ratio=0.59, non_text_count=2)

        self.assertNotEqual(classify_region(full_page).region_type, "two_column")
        self.assertEqual(classify_region(content_band).region_type, "two_column")

    def test_two_column_requires_vertical_partition_to_dominate(self):
        horizontal_band = make_region_features(
            page_height_ratio=0.64,
            strongest_vertical_gap=0.25,
            strongest_horizontal_separator=0.58,
            text_ratio=0.6,
            non_text_count=2,
        )

        self.assertNotEqual(classify_region(horizontal_band).region_type, "two_column")


if __name__ == "__main__":
    unittest.main()
