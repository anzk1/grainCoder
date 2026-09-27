import unittest

from scripts.summarize_segmentation_ablation import generation_summary, paired_summary


class AblationSummaryTest(unittest.TestCase):
    def test_paired_metrics(self):
        scores = {
            "b0_dcgen_segmentation": {"0": 0.5, "1": 0.6, "2": 0.7},
            "b1_layoutcoder_mvp_segmentation": {"0": 0.52, "1": 0.59, "2": 0.7},
        }
        summary = paired_summary(scores)
        self.assertEqual(summary["paired"]["wins"], 1)
        self.assertEqual(summary["paired"]["losses"], 1)
        self.assertEqual(summary["paired"]["ties"], 1)
        self.assertEqual(summary["paired"]["improvement_gt_0_01"], 1)

    def test_generation_metrics(self):
        records = [
            {"status": "ok", "segmenter": "b1", "elapsed_seconds": 2, "segmentation": {"leaf_count": 1, "tree_depth": 2}},
            {"status": "failed", "segmenter": "b1", "elapsed_seconds": 1},
        ]
        summary = generation_summary(records)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["by_segmenter"]["b1"]["single_leaf_count"], 1)


if __name__ == "__main__":
    unittest.main()
