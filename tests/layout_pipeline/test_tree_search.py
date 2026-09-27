import unittest
from unittest.mock import patch

import numpy as np

from layout_pipeline.page_mask import RegionCandidateSet
from layout_pipeline.schema import RegionDecisionCandidate, RegionProfile, SeparatorLine, SplitCandidate
from layout_pipeline.tree_search import search_layout_trees
from tests.layout_pipeline.test_region_classifier import make_region_features


def no_split_decision(score: float) -> RegionDecisionCandidate:
    return RegionDecisionCandidate("no_split", None, score, 0.0, score, ("test_no_split",))


def split_decision(score: float, reasons=()) -> RegionDecisionCandidate:
    candidate = SplitCandidate(
        (0, 0, 100, 100),
        SeparatorLine("vertical", 50, 0, 100, "whitespace"),
        ("whitespace",),
        score,
        True,
        (),
        2,
        2,
        (),
        (),
        20,
        1.0,
    )
    return RegionDecisionCandidate("split", candidate, score, 0.0, score, tuple(reasons))


class TreeSearchTest(unittest.TestCase):
    def test_no_split_can_beat_higher_local_separator(self):
        root_profile = RegionProfile(
            "text_section",
            0.9,
            make_region_features(text_ratio=1.0, non_text_count=0, background_continuity=0.95),
            ("text_dominant",),
        )
        child_profile = RegionProfile("text_section", 0.8, root_profile.features, ("text_dominant",))
        root_candidates = RegionCandidateSet(root_profile, (), (no_split_decision(0.7), split_decision(0.9, ("text_fragment",))))
        child_candidates = RegionCandidateSet(child_profile, (), (no_split_decision(0.6),))

        with patch(
            "layout_pipeline.tree_search.build_region_candidate_set",
            side_effect=lambda region, *args, **kwargs: root_candidates if region == (0, 0, 100, 100) else child_candidates,
        ):
            outcome = search_layout_trees(
                np.zeros((100, 100, 3), dtype=np.float32),
                (100, 100),
                [],
                [],
                [],
                {"mixed": {}},
                {"beam_width": 4, "max_split_candidates_per_region": 3, "max_expansions": 8, "max_depth": 3},
                {"semantic_cohesion": 1.0, "generation_cost_penalty": 0.5},
                8,
                10,
                0.55,
                0.08,
            )

        self.assertFalse(outcome.ranked_states[0].split_tree.children)
        self.assertTrue(any(state.split_tree.children for state in outcome.ranked_states[1:]))

    def test_depth_guard_finalizes_children_without_recursive_explosion(self):
        profile = RegionProfile("mixed", 0.7, make_region_features(), ("mixed",))
        candidate_set = RegionCandidateSet(profile, (), (no_split_decision(0.1), split_decision(0.9)))

        with patch("layout_pipeline.tree_search.build_region_candidate_set", return_value=candidate_set):
            outcome = search_layout_trees(
                np.zeros((100, 100, 3), dtype=np.float32),
                (100, 100),
                [],
                [],
                [],
                {"mixed": {}},
                {"beam_width": 4, "max_split_candidates_per_region": 3, "max_expansions": 8, "max_depth": 1},
                {"separator_evidence": 1.0},
                8,
                10,
                0.55,
                0.08,
            )

        split_state = next(state for state in outcome.ranked_states if state.split_tree.children)
        self.assertTrue(all(not child.children for child in split_state.split_tree.children))
        self.assertTrue(any("max_depth_guard" in trace.decision.reasons for trace in split_state.decision_traces))

    def test_partial_local_score_keeps_promising_split_branch_in_beam(self):
        profile = RegionProfile("mixed", 0.7, make_region_features(), ("mixed",))
        candidate_set = RegionCandidateSet(profile, (), (no_split_decision(-0.7), split_decision(1.1)))

        with patch("layout_pipeline.tree_search.build_region_candidate_set", return_value=candidate_set):
            outcome = search_layout_trees(
                np.zeros((100, 100, 3), dtype=np.float32),
                (100, 100),
                [],
                [],
                [],
                {"mixed": {}},
                {
                    "beam_width": 2,
                    "max_split_candidates_per_region": 1,
                    "max_expansions": 3,
                    "max_depth": 1,
                    "partial_local_score_weight": 0.1,
                },
                {"generation_cost_penalty": 1.0},
                8,
                10,
                0.55,
                0.08,
            )

        self.assertTrue(any(state.split_tree.children for state in outcome.ranked_states))


if __name__ == "__main__":
    unittest.main()
