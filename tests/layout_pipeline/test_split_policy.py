import unittest

from layout_pipeline.region_classifier import classify_region
from layout_pipeline.schema import ElementNode, LayoutGroup, SeparatorLine, SplitCandidate
from layout_pipeline.split_policy import build_region_decisions
from tests.layout_pipeline.test_region_classifier import make_region_features


def split_candidate(direction: str, position: int, score: float, sources=("whitespace",)) -> SplitCandidate:
    line = SeparatorLine(direction, position, 0, 100, sources[0])
    return SplitCandidate((0, 0, 100, 100), line, sources, score, True, (), 2, 2, (), (), 20, 1.0)


class SplitPolicyTest(unittest.TestCase):
    def test_includes_no_split_and_deduplicates_nearby_axis_candidates(self):
        profile = classify_region(make_region_features(strongest_vertical_gap=0.8, column_group_count=2))
        candidates = [
            split_candidate("vertical", 48, 0.8),
            split_candidate("vertical", 52, 0.9),
            split_candidate("horizontal", 50, 0.7),
        ]

        decisions = build_region_decisions(
            profile,
            candidates,
            [],
            [],
            {"no_split_bias": 0.05, "minimum_split_score": 0.6, "preferred_direction_bonus": 0.5, "preferred_directions": ["vertical"]},
        )

        vertical = [decision for decision in decisions if decision.split_candidate and decision.split_candidate.line.direction == "vertical"]
        self.assertEqual(len(vertical), 1)
        self.assertEqual(vertical[0].split_candidate.line.position, 52)
        self.assertTrue(any(decision.decision_type == "no_split" for decision in decisions))
        self.assertIn("preferred_direction", vertical[0].reasons)

    def test_hero_policy_penalizes_weak_large_component_cut(self):
        profile = classify_region(
            make_region_features(page_area_ratio=0.8, largest_component_area_ratio=0.6, text_count=1)
        )
        candidate = SplitCandidate(
            (0, 0, 100, 100),
            SeparatorLine("vertical", 50, 0, 100, "whitespace"),
            ("whitespace",),
            0.9,
            True,
            (),
            1,
            1,
            (0,),
            (),
            20,
            1.0,
            large_component_crossing_ids=(0,),
        )
        elements = [ElementNode(0, (10, 10, 90, 90), "Compo")]

        decisions = build_region_decisions(
            profile,
            [candidate],
            elements,
            [],
            {"no_split_bias": 0.4, "minimum_split_score": 0.95, "large_component_cut_penalty": 0.8},
        )

        split = next(decision for decision in decisions if decision.decision_type == "split")
        self.assertIn("large_component_cut", split.reasons)
        self.assertLess(split.final_local_score, 0.95)

    def test_layout_group_source_does_not_exempt_crossing_another_group(self):
        profile = classify_region(
            make_region_features(
                foreground_count=8,
                row_group_count=2,
                repeated_size_score=0.9,
                repeated_gap_score=0.9,
            )
        )
        candidate = split_candidate("horizontal", 50, 1.0, ("layout_group",))
        crossing_group = LayoutGroup(4, "row", (0, 20, 100, 80), (0, 1), "Text")

        decisions = build_region_decisions(
            profile,
            [candidate],
            [],
            [crossing_group],
            {
                "no_split_bias": 0.05,
                "minimum_split_score": 0.6,
                "layout_group_bonus": 0.45,
                "repeated_unit_bonus": 0.4,
                "broken_layout_group_penalty": 0.8,
            },
        )

        split = next(decision for decision in decisions if decision.decision_type == "split")
        self.assertIn("broken_layout_group", split.reasons)

    def test_policy_selects_safe_component_edge_over_higher_raw_cut(self):
        profile = classify_region(
            make_region_features(
                page_height_ratio=0.59,
                foreground_count=5,
                strongest_vertical_gap=0.25,
                strongest_horizontal_gap=0.2,
            )
        )
        unsafe = SplitCandidate(
            (0, 0, 200, 100),
            SeparatorLine("vertical", 120, 0, 100, "whitespace"),
            ("whitespace",),
            0.9,
            True,
            (),
            3,
            2,
            (),
            (2,),
            20,
            0.67,
            large_component_crossing_ids=(2,),
        )
        safe = SplitCandidate(
            (0, 0, 200, 100),
            SeparatorLine("vertical", 100, 0, 100, "layout_group"),
            ("layout_group",),
            0.5,
            True,
            (),
            3,
            2,
            (),
            (),
            0,
            0.67,
        )

        decisions = build_region_decisions(
            profile,
            [unsafe, safe],
            [],
            [],
            {
                "no_split_bias": 0.05,
                "minimum_split_score": 0.6,
                "large_component_cut_penalty": 0.8,
                "preferred_direction_bonus": 0.5,
                "preferred_directions": ["vertical"],
            },
        )

        vertical = next(decision for decision in decisions if decision.split_candidate and decision.split_candidate.line.direction == "vertical")
        self.assertEqual(vertical.split_candidate.line.position, 100)

if __name__ == "__main__":
    unittest.main()
