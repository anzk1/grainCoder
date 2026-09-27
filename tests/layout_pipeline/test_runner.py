import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_uied_graph_layout_pipeline import (
    ApiCapacityGate,
    _ask_with_capacity_retry,
    _load_passing_structure,
    _load_matching_candidate_checkpoint,
    _select_atomic_snippet,
    compare_results,
    parse_args,
    run_candidate,
)


class RunnerTest(unittest.TestCase):
    def test_help_exposes_structure_operation(self):
        with patch("sys.argv", ["runner", "structure", "--help"]):
            with self.assertRaises(SystemExit) as raised:
                parse_args()
        self.assertEqual(raised.exception.code, 0)

    def test_missing_validation_blocks_candidate_before_model_access(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(RuntimeError, "no persisted structure validation"):
                _load_passing_structure(temporary_directory, "18")

            args = argparse.Namespace(
                generation_workers=1,
                config="configs/uied_graph_layout_tree.yaml",
                output_dir=temporary_directory,
                ids=["18"],
            )
            with self.assertRaisesRegex(RuntimeError, "no persisted structure validation"):
                run_candidate(args)

    def test_old_structure_without_segmentation_role_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            sample_root = Path(temporary_directory) / "candidate_results" / "samples" / "18"
            sample_root.mkdir(parents=True)
            (sample_root / "validation.json").write_text('{"passed":true}', encoding="utf-8")
            (sample_root / "structure.json").write_text(
                '{"id":0,"type":"atomic","bbox":[0,0,100,100],"element_ids":[0]}',
                encoding="utf-8",
            )
            (sample_root / "merged_uied.json").write_text(
                '{"elements":[{"id":0,"bbox":[0,0,10,10],"class":"Text","text":"Title"}]}',
                encoding="utf-8",
            )

            with self.assertRaises(KeyError):
                _load_passing_structure(temporary_directory, "18")

    def test_capacity_error_retries_only_the_failed_request(self):
        class CapacityBot:
            def __init__(self):
                self.calls = 0

            def ask(self, prompt, encoded_crop):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("exceed limit")
                return "ok"

        bot = CapacityBot()

        response, request_count = _ask_with_capacity_retry(bot, "prompt", "image", ApiCapacityGate(cooldown_seconds=0))

        self.assertEqual(response, "ok")
        self.assertEqual(bot.calls, 2)
        self.assertEqual(request_count, 2)

    def test_checkpoint_requires_matching_generation_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.json"
            checkpoint_path.write_text(
                '{"identity":{"structure_hash":"current"},"snippets":{"7":"<div>ok</div>"},"traces":[{"atomic_id":7}]}',
                encoding="utf-8",
            )

            snippets, traces = _load_matching_candidate_checkpoint(
                checkpoint_path,
                {"structure_hash": "current"},
            )
            stale_snippets, stale_traces = _load_matching_candidate_checkpoint(
                checkpoint_path,
                {"structure_hash": "changed"},
            )

        self.assertEqual(snippets, {7: "<div>ok</div>"})
        self.assertEqual(traces, [{"atomic_id": 7}])
        self.assertEqual(stale_snippets, {})
        self.assertEqual(stale_traces, [])

    def test_rejected_initial_candidates_trigger_one_replacement(self):
        from PIL import Image

        from layout_pipeline.semantic_renderer import SemanticLayoutRenderer

        class CandidateBot:
            def __init__(self):
                self.responses = iter(("```css bad ```", "```css bad ```", "<div>valid</div>"))

            def ask(self, prompt, encoded_crop):
                return next(self.responses)

        crop = Image.new("RGB", (10, 10), "white")
        with patch(
            "scripts.run_uied_graph_layout_pipeline._render_html",
            return_value=Image.new("RGB", (10, 10), "white"),
        ):
            selected, trace, request_count = _select_atomic_snippet(
                CandidateBot(),
                "prompt",
                "image",
                crop,
                object(),
                SemanticLayoutRenderer(),
                ApiCapacityGate(cooldown_seconds=0),
                2,
            )

        self.assertEqual(selected, "<div>valid</div>")
        self.assertEqual(request_count, 3)
        self.assertEqual(trace["selected_candidate"], 2)
        self.assertEqual([candidate["accepted"] for candidate in trace["candidates"]], [False, False, True])

    def test_compare_uses_frozen_scored_split_acceptance_contract(self):
        sample_ids = ["0", "12", "13", "18", "19", "22", "25", "28", "29", "49", "91", "95"]
        baseline_scores = {sample_id: 0.94 for sample_id in sample_ids}
        candidate_scores = {sample_id: 0.95 for sample_id in sample_ids}
        candidate_scores["22"] = 0.965
        candidate_scores["91"] = 0.975
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frozen_path = root / "frozen.json"
            frozen_path.write_text(json.dumps({"layoutcoder_structure_flex": baseline_scores}), encoding="utf-8")
            candidate_root = root / "candidate_results"
            candidate_root.mkdir()
            (candidate_root / "final_clip_scores.json").write_text(
                json.dumps({"layoutcoder_structure_flex": candidate_scores}),
                encoding="utf-8",
            )
            (candidate_root / "generation_records.json").write_text(
                json.dumps([{"sample_id": sample_id, "status": "ok", "actual_api_calls": 30} for sample_id in sample_ids]),
                encoding="utf-8",
            )
            with patch("scripts.run_uied_graph_layout_pipeline.FROZEN_SCORE_PATH", frozen_path):
                comparison = compare_results(temporary_directory)

        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["actual_api_calls"], 360)


if __name__ == "__main__":
    unittest.main()
