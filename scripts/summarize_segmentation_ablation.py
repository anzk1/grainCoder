#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import math
import statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize paired segmentation ablation records.")
    parser.add_argument("--raw-scores", required=True, help="JSON mapping method -> sample -> CLIP score")
    parser.add_argument("--final-scores", required=True, help="JSON mapping method -> sample -> CLIP score")
    parser.add_argument("--generation-records", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def descriptive(values):
    if not values:
        return {"sample_count": 0, "mean": None, "median": None, "min": None, "p10": None, "bottom_10_mean": None}
    bottom_count = min(10, len(values))
    return {
        "sample_count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "p10": percentile(values, 0.10),
        "bottom_10_mean": statistics.fmean(sorted(values)[:bottom_count]),
    }


def paired_summary(score_payload, baseline="b0_dcgen_segmentation", candidate="b1_layoutcoder_mvp_segmentation"):
    baseline_scores = score_payload.get(baseline, {})
    candidate_scores = score_payload.get(candidate, {})
    sample_ids = sorted(set(baseline_scores) & set(candidate_scores))
    deltas = [candidate_scores[sample_id] - baseline_scores[sample_id] for sample_id in sample_ids]
    return {
        "baseline": descriptive([baseline_scores[sample_id] for sample_id in sample_ids]),
        "candidate": descriptive([candidate_scores[sample_id] for sample_id in sample_ids]),
        "paired": {
            "sample_count": len(deltas),
            "wins": sum(delta > 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "mean_delta": statistics.fmean(deltas) if deltas else None,
            "median_delta": statistics.median(deltas) if deltas else None,
            "improvement_gt_0_01": sum(delta > 0.01 for delta in deltas),
            "regression_lt_minus_0_01": sum(delta < -0.01 for delta in deltas),
            "regression_lt_minus_0_03": sum(delta < -0.03 for delta in deltas),
            "deltas": dict(zip(sample_ids, deltas)),
        },
    }


def generation_summary(records):
    successful = [record for record in records if record.get("status") == "ok"]
    times = [record["elapsed_seconds"] for record in successful if "elapsed_seconds" in record]
    failed = [record for record in records if record.get("status") == "failed"]
    by_segmenter = {}
    for record in successful:
        by_segmenter.setdefault(record.get("segmenter", "unknown"), []).append(record)
    return {
        "record_count": len(records),
        "failure_count": len(failed),
        "failure_rate": len(failed) / len(records) if records else None,
        "mean_elapsed_seconds": statistics.fmean(times) if times else None,
        "by_segmenter": {
            name: {
                "record_count": len(items),
                "mean_elapsed_seconds": statistics.fmean([item["elapsed_seconds"] for item in items]),
                "mean_api_calls": _mean_top_level(items, "estimated_api_calls"),
                "total_api_calls": sum(item.get("estimated_api_calls", 0) for item in items),
                "mean_leaf_count": _mean_nested(items, "leaf_count"),
                "mean_tree_depth": _mean_nested(items, "tree_depth"),
                "mean_uied_box_count": _mean_nested(items, "uied_box_count"),
                "mean_normalized_box_count": _mean_nested(items, "normalized_box_count"),
                "single_leaf_count": sum(item.get("segmentation", {}).get("leaf_count") == 1 for item in items),
                "reached_max_leaves_count": sum(bool(item.get("segmentation", {}).get("reached_max_leaves")) for item in items),
                "reached_max_depth_count": sum(bool(item.get("segmentation", {}).get("reached_max_depth")) for item in items),
            }
            for name, items in by_segmenter.items()
        },
    }


def _mean_top_level(records, key):
    values = [record.get(key) for record in records]
    values = [value for value in values if isinstance(value, (int, float))]
    return statistics.fmean(values) if values else None


def _mean_nested(records, key):
    values = [record.get("segmentation", {}).get(key) for record in records]
    values = [value for value in values if isinstance(value, (int, float))]
    return statistics.fmean(values) if values else None


def main():
    args = parse_args()
    raw_scores = json.loads(Path(args.raw_scores).read_text(encoding="utf-8"))
    final_scores = json.loads(Path(args.final_scores).read_text(encoding="utf-8"))
    records = json.loads(Path(args.generation_records).read_text(encoding="utf-8"))
    summary = {
        "raw_clip": paired_summary(raw_scores),
        "final_clip": paired_summary(final_scores),
        "generation_and_segmentation": generation_summary(records),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
