#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json

from segmentation.layoutcoder_mvp.config import load_config, pipeline_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect LayoutCoder-style UIED projection segmentation.")
    parser.add_argument("images", nargs="+", help="Screenshot paths")
    parser.add_argument("--output-dir", default="debug/layoutcoder_mvp")
    parser.add_argument("--config", default="configs/layoutcoder_segmentation_mvp.yaml")
    parser.add_argument("--layoutcoder-root", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = pipeline_from_config(load_config(args.config), args.layoutcoder_root)
    records = []
    for image_path in args.images:
        sample_id = Path(image_path).stem
        output_dir = Path(args.output_dir) / sample_id
        try:
            run = pipeline.segment(image_path, str(output_dir))
            records.append({"sample_id": sample_id, "status": "ok", "elapsed_seconds": run.elapsed_seconds})
        except Exception as exc:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "error.txt").write_text(str(exc), encoding="utf-8")
            records.append({"sample_id": sample_id, "status": "failed", "error": str(exc)})
    destination = Path(args.output_dir) / "inspection_records.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2), encoding="utf-8")
    if any(record["status"] == "failed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
