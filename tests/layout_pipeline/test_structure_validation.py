import unittest
from dataclasses import replace

from layout_pipeline.schema import ElementNode, LayoutNode, RegionProfile, SeparatorLine, SplitCandidate
from layout_pipeline.structure_validation import validate_structure
from tests.layout_pipeline.test_region_classifier import make_region_features


class StructureValidationTest(unittest.TestCase):
    def test_rejects_overlapping_siblings(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 100),
            "row",
            (
                LayoutNode(1, (0, 0, 60, 100), "atomic", element_ids=(0,), ownership_label="content"),
                LayoutNode(2, (50, 0, 100, 100), "atomic", element_ids=(1,), ownership_label="content"),
            ),
        )
        elements = [ElementNode(0, (5, 5, 20, 20), "Text"), ElementNode(1, (70, 5, 90, 20), "Text")]

        validation = validate_structure(root, elements)

        self.assertFalse(validation.passed)
        self.assertTrue(any("overlap" in error for error in validation.errors))

    def test_reports_atomic_quality_ratios(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 100),
            "row",
            (
                LayoutNode(1, (0, 0, 50, 100), "atomic", element_ids=(0,), ownership_label="content"),
                LayoutNode(2, (50, 0, 100, 100), "atomic", element_ids=(1,), ownership_label="content"),
            ),
        )
        elements = [ElementNode(0, (5, 5, 20, 20), "Text"), ElementNode(1, (70, 5, 90, 20), "Compo")]
        candidate = SplitCandidate(
            (0, 0, 100, 100),
            SeparatorLine("vertical", 50, 0, 100, "whitespace"),
            ("whitespace",),
            0.7,
            True,
            (),
            1,
            1,
            (),
            (),
            30,
            1.0,
            selected=True,
        )

        validation = validate_structure(root, elements, [candidate], strong_separator_score=0.9)

        self.assertEqual(validation.largest_atomic_element_ratio, 0.5)
        self.assertEqual(validation.thin_text_atomic_ratio, 0.5)
        self.assertEqual(validation.single_element_atomic_ratio, 1.0)
        self.assertEqual(validation.weak_separator_ratio, 1.0)

    def test_quality_ratios_fail_structure_gate(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 100),
            "row",
            (
                LayoutNode(1, (0, 0, 50, 100), "atomic", element_ids=(0,), ownership_label="content"),
                LayoutNode(2, (50, 0, 100, 100), "atomic", ownership_label="content"),
            ),
        )
        elements = [ElementNode(0, (5, 5, 20, 15), "Text", "line")]

        validation = validate_structure(root, elements)

        self.assertFalse(validation.passed)
        self.assertEqual(validation.empty_atomic_ratio, 0.5)
        self.assertEqual(validation.small_single_element_ratio, 1.0)
        self.assertTrue(any("empty_atomic_ratio" in error for error in validation.errors))

    def test_layout_only_spacer_does_not_count_as_empty_atomic(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 100),
            "row",
            (
                LayoutNode(1, (0, 0, 50, 100), "spacer"),
                LayoutNode(2, (50, 0, 100, 100), "atomic", element_ids=(0,), ownership_label="content"),
            ),
        )
        elements = [ElementNode(0, (55, 10, 95, 90), "Compo")]

        validation = validate_structure(root, elements)

        self.assertTrue(validation.passed)
        self.assertEqual(validation.empty_atomic_ratio, 0.0)

    def test_rejects_under_segmented_mixed_atomic_with_internal_boundaries(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 100),
            "atomic",
            element_ids=tuple(range(8)),
            ownership_label="content",
            region_type="mixed",
        )
        elements = [ElementNode(index, (5, 5 + index * 10, 95, 12 + index * 10), "Text", str(index)) for index in range(8)]
        profile = RegionProfile(
            "mixed",
            0.8,
            make_region_features(
                bbox=(0, 0, 100, 100),
                foreground_count=8,
                strongest_horizontal_separator=0.9,
                row_group_count=2,
            ),
            ("mixed",),
        )

        validation = validate_structure(root, elements, region_profiles=[profile])

        self.assertFalse(validation.passed)
        self.assertTrue(any("internal partition evidence" in error for error in validation.errors))

    def test_rejects_two_column_region_left_as_one_atomic(self):
        root = LayoutNode(0, (0, 0, 100, 100), "atomic", element_ids=(0, 1, 2, 3), region_type="two_column")
        elements = [ElementNode(index, (5 + index * 20, 10, 20 + index * 20, 90), "Compo") for index in range(4)]
        profile = RegionProfile(
            "two_column",
            0.8,
            make_region_features(
                bbox=(0, 0, 100, 100),
                foreground_count=4,
                strongest_vertical_gap=0.4,
            ),
            ("vertical_partition",),
        )

        validation = validate_structure(root, elements, region_profiles=[profile])

        self.assertFalse(validation.passed)
        self.assertTrue(any("remains unsplit" in error for error in validation.errors))

    def test_rejects_text_and_container_atomic_with_component_boundary(self):
        root = LayoutNode(
            0,
            (0, 0, 100, 100),
            "atomic",
            element_ids=(0, 1, 2),
            ownership_label="content",
            region_type="text_section",
        )
        elements = [
            ElementNode(0, (5, 10, 40, 30), "Text", "first"),
            ElementNode(1, (5, 40, 40, 60), "Text", "second"),
            ElementNode(2, (50, 5, 95, 95), "Block", segmentation_role="container"),
        ]
        profile = RegionProfile(
            "text_section",
            0.8,
            make_region_features(bbox=(0, 0, 100, 100), foreground_count=2, text_count=2),
            ("text_dominant",),
        )
        component_boundary = SplitCandidate(
            (0, 0, 100, 100),
            SeparatorLine("vertical", 50, 0, 100, "component_boundary"),
            ("component_boundary",),
            0.9,
            True,
            (),
            2,
            0,
            (),
            (),
            0,
            0.5,
        )

        validation = validate_structure(
            root,
            elements,
            split_candidates=[component_boundary],
            region_profiles=[profile],
        )

        self.assertFalse(validation.passed)
        self.assertTrue(any("accepted vertical component boundary" in error for error in validation.errors))

        stacked_section = validate_structure(
            root,
            elements,
            split_candidates=[
                replace(
                    component_boundary,
                    line=SeparatorLine("horizontal", 50, 0, 100, "component_boundary"),
                )
            ],
            region_profiles=[profile],
        )

        self.assertTrue(stacked_section.passed)


if __name__ == "__main__":
    unittest.main()
