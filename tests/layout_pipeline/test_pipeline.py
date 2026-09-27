import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from layout_pipeline.page_mask import PageMaskBuild
from layout_pipeline.pipeline import LayoutCoderStructurePipeline, _merged_elements, _segmentation_role
from layout_pipeline.schema import RegionDecisionCandidate, RegionProfile, SeparatorLine, SplitCandidate, SplitNode
from layout_pipeline.tree_scoring import TreeScoreBreakdown
from layout_pipeline.tree_search import RegionDecisionTrace, TreeSearchState
from tests.layout_pipeline.test_region_classifier import make_region_features
from segmentation.layoutcoder_mvp.uied_backend import UIEDDetectionArtifacts


class PipelineTest(unittest.TestCase):
    def test_merged_payload_is_the_canonical_scaled_element_source(self):
        artifacts = UIEDDetectionArtifacts(
            image_width=200,
            image_height=100,
            ip_payload={"img_shape": [50, 100, 3], "compos": []},
            ocr_payload={"img_shape": [50, 100, 3], "texts": []},
            merge_payload={
                "img_shape": [50, 100, 3],
                "compos": [
                    {
                        "class": "Text",
                        "text_content": "Hello",
                        "position": {"column_min": 10, "row_min": 5, "column_max": 40, "row_max": 20},
                    }
                ],
            },
        )

        elements = _merged_elements(artifacts)

        self.assertEqual(elements[0].bbox, (20, 10, 80, 40))
        self.assertEqual(elements[0].text, "Hello")

    def test_classifies_page_scale_boxes_outside_foreground(self):
        self.assertEqual(_segmentation_role("Text", (0, 0, 200, 20), (200, 100), "Title"), "foreground")
        self.assertEqual(_segmentation_role("Compo", (0, 80, 200, 100), (200, 100), None), "container")
        self.assertEqual(_segmentation_role("Block", (0, 0, 200, 100), (200, 100), None), "background")

    @patch("layout_pipeline.pipeline.build_layout_groups", return_value=[])
    @patch("layout_pipeline.pipeline.detect_separator_lines", return_value=[])
    @patch("layout_pipeline.pipeline.build_page_mask")
    def test_retries_single_atomic_without_container_boxes(self, build_page_mask, detect_lines, build_groups):
        first_build = PageMaskBuild(
            Image.new("RGB", (200, 100)),
            (),
            SplitNode((0, 0, 200, 100)),
            (),
            0.9,
        )
        split_line = SeparatorLine("vertical", 80, 0, 100, "whitespace")
        second_build = PageMaskBuild(
            Image.new("RGB", (200, 100)),
            (split_line,),
            SplitNode(
                (0, 0, 200, 100),
                (SplitNode((0, 0, 80, 100)), SplitNode((80, 0, 200, 100))),
                split_line,
                0.8,
                ("whitespace",),
            ),
            (),
            0.9,
        )
        build_page_mask.side_effect = [first_build, second_build]
        components = [
            {
                "class": "Text",
                "text_content": f"Item {index}",
                "position": {
                    "column_min": 10 if index < 4 else 120,
                    "row_min": 10 + index * 8,
                    "column_max": 30 if index < 4 else 140,
                    "row_max": 16 + index * 8,
                },
            }
            for index in range(8)
        ]
        components.append(
            {
                "class": "Block",
                "position": {"column_min": 0, "row_min": 0, "column_max": 200, "row_max": 100},
            }
        )
        artifacts = UIEDDetectionArtifacts(
            image_width=200,
            image_height=100,
            ip_payload={"img_shape": [100, 200, 3], "compos": []},
            ocr_payload={"img_shape": [100, 200, 3], "texts": []},
            merge_payload={"img_shape": [100, 200, 3], "compos": components},
        )
        pipeline = LayoutCoderStructurePipeline.__new__(LayoutCoderStructurePipeline)
        pipeline.config = {"mask": {"second_pass_min_foreground_elements": 8}, "ownership": {"icon_max_size": 48}}

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "sample.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            analysis = pipeline.analyze_artifacts(str(image_path), artifacts)

        self.assertTrue(analysis.second_pass_used)
        self.assertEqual(build_page_mask.call_count, 2)
        second_pass_elements = build_page_mask.call_args_list[1].args[1]
        self.assertTrue(second_pass_elements)
        self.assertTrue(all(element.segmentation_role == "foreground" for element in second_pass_elements))

    @patch("layout_pipeline.pipeline.build_layout_groups", return_value=[])
    @patch("layout_pipeline.pipeline.detect_separator_lines", return_value=[])
    @patch("layout_pipeline.pipeline.build_page_mask")
    def test_selects_next_ranked_tree_when_top_tree_fails_quality_gate(self, build_page_mask, detect_lines, build_groups):
        split_line = SeparatorLine("vertical", 50, 0, 100, "whitespace")
        split_tree = SplitNode(
            (0, 0, 100, 100),
            (SplitNode((0, 0, 50, 100)), SplitNode((50, 0, 100, 100))),
            split_line,
            0.8,
            ("whitespace",),
        )
        no_split_tree = SplitNode((0, 0, 100, 100))
        breakdown = TreeScoreBreakdown(0.5, {}, {})
        split_candidate = SplitCandidate(
            (0, 0, 100, 100),
            split_line,
            ("whitespace",),
            0.8,
            True,
            (),
            1,
            1,
            (),
            (),
            20,
            1.0,
        )
        hero_profile = RegionProfile(
            "hero",
            0.8,
            make_region_features(page_area_ratio=1.0, largest_component_area_ratio=0.8),
            ("large_component",),
        )
        trace = RegionDecisionTrace(
            hero_profile,
            RegionDecisionCandidate("split", split_candidate, 0.8, 0.0, 0.8, ()),
            0,
        )
        top_state = TreeSearchState(split_tree, (), 0.8, 0.8, 1, (trace,), breakdown)
        fallback_state = TreeSearchState(no_split_tree, (), 0.5, 0.5, 1, (), breakdown)
        build_page_mask.return_value = PageMaskBuild(
            Image.new("RGB", (100, 100)),
            (split_line,),
            split_tree,
            (),
            0.9,
            (top_state, fallback_state),
        )
        artifacts = UIEDDetectionArtifacts(
            image_width=100,
            image_height=100,
            ip_payload={"img_shape": [100, 100, 3], "compos": []},
            ocr_payload={"img_shape": [100, 100, 3], "texts": []},
            merge_payload={
                "img_shape": [100, 100, 3],
                "compos": [
                    {
                        "class": "Compo",
                        "position": {"column_min": 10, "row_min": 10, "column_max": 90, "row_max": 90},
                    }
                ],
            },
        )
        pipeline = LayoutCoderStructurePipeline.__new__(LayoutCoderStructurePipeline)
        pipeline.config = {"mask": {}, "ownership": {"icon_max_size": 48}, "quality_gate": {}}

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "sample.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            analysis = pipeline.analyze_artifacts(str(image_path), artifacts)

        self.assertTrue(analysis.validation.passed)
        self.assertEqual(analysis.chosen_tree_rank, 1)
        self.assertEqual(len(analysis.rejected_trees), 1)


if __name__ == "__main__":
    unittest.main()
