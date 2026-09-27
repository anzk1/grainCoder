#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI
from PIL import Image

from segmentation.layoutcoder_mvp.config import load_config, pipeline_from_config


DIAGNOSTIC_IDS = ["0", "12", "13", "18", "19", "22", "25", "28", "29", "49", "91", "95"]
BASELINE_NAME = "b0_dcgen_segmentation"
LAYOUTCODER_NAME = "b1_layoutcoder_mvp_segmentation"
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL")
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run paired DCGen segmentation ablation.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ids", nargs="*", default=DIAGNOSTIC_IDS)
    parser.add_argument("--config", default="configs/layoutcoder_segmentation_mvp.yaml")
    parser.add_argument("--layoutcoder-root", required=True)
    parser.add_argument("--key-path", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--generation-workers", type=int, default=24)
    parser.add_argument("--segmentation-workers", type=int, default=2)
    parser.add_argument("--leaf-parallel", action="store_true")
    parser.add_argument("--render-and-evaluate", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def build_baseline_tree(image_path, baseline_config):
    from utils import ImgSegmentation

    return ImgSegmentation(str(image_path), **baseline_config)


def tree_stats(tree):
    payload = tree.to_json_tree()

    def count_leaves(node):
        return 1 if not node["children"] else sum(count_leaves(child) for child in node["children"])

    def depth(node):
        return 1 if not node["children"] else 1 + max(depth(child) for child in node["children"])

    return {"leaf_count": count_leaves(payload), "tree_depth": depth(payload)}


def create_bot(key_path, model, base_url):
    from utils import GPT4

    bot = GPT4(key_path=key_path, model=model)
    client_options = {
        "api_key": bot.key,
        "default_headers": BROWSER_HEADERS,
    }
    if base_url:
        client_options["base_url"] = base_url
    bot.client = OpenAI(**client_options)
    return bot


def generate_dcgen(tree, bot, leaf_parallel):
    from scripts.experiments import prompt_dcgen
    from utils import DCGenGrid

    grid = DCGenGrid(tree, prompt_seg=prompt_dcgen["prompt_leaf"], prompt_refine=prompt_dcgen["prompt_root"])
    final_html = grid.generate_code(bot, multi_thread=leaf_parallel)
    if not grid.raw_code or not final_html:
        raise RuntimeError("DCGen returned empty raw or final HTML")
    return grid.raw_code, final_html


def prepare_segmentation_tasks(args, config, output_root):
    image_dir = Path(args.image_dir)
    tasks = []
    missing_records = []
    for sample_id in args.ids:
        image_path = image_dir / f"{sample_id}.png"
        if not image_path.is_file():
            missing_records.append({"sample_id": sample_id, "status": "missing_image"})
            continue
        baseline_tree = build_baseline_tree(image_path, config["baseline"])
        tasks.append(
            {
                "sample_id": sample_id,
                "segmenter": BASELINE_NAME,
                "image_path": image_path,
                "tree": baseline_tree,
                "segmentation": tree_stats(baseline_tree),
            }
        )

    pipeline = pipeline_from_config(config, args.layoutcoder_root)

    def prepare_layoutcoder(sample_id):
        image_path = image_dir / f"{sample_id}.png"
        run = pipeline.segment(str(image_path), str(output_root / "segmentation" / sample_id))
        stats = json.loads(
            (output_root / "segmentation" / sample_id / "segmentation_stats.json").read_text(encoding="utf-8")
        )
        return {
            "sample_id": sample_id,
            "segmenter": LAYOUTCODER_NAME,
            "image_path": image_path,
            "tree": run.dcgen_root,
            "segmentation": stats,
        }

    available_ids = [sample_id for sample_id in args.ids if (image_dir / f"{sample_id}.png").is_file()]
    with ThreadPoolExecutor(max_workers=args.segmentation_workers) as executor:
        future_map = {executor.submit(prepare_layoutcoder, sample_id): sample_id for sample_id in available_ids}
        for future in as_completed(future_map):
            sample_id = future_map[future]
            try:
                tasks.append(future.result())
            except Exception as exc:
                missing_records.append(
                    {"sample_id": sample_id, "segmenter": LAYOUTCODER_NAME, "status": "failed", "error": str(exc)}
                )
    tasks.sort(key=lambda task: (task["sample_id"], task["segmenter"]))
    return tasks, missing_records


def run_generation_task(task, args, config, output_root):
    sample_id = task["sample_id"]
    segmenter_name = task["segmenter"]
    method_dir = output_root / "outputs" / segmenter_name
    method_dir.mkdir(parents=True, exist_ok=True)
    raw_path = method_dir / f"{sample_id}.raw.html"
    final_path = method_dir / f"{sample_id}.html"
    record = {
        "sample_id": sample_id,
        "segmenter": segmenter_name,
        "segmentation": task["segmentation"],
        "thread_name": threading.current_thread().name,
    }
    started_at = perf_counter()
    try:
        if raw_path.is_file() and final_path.is_file() and not args.replace:
            record["generation_status"] = "reused"
        else:
            bot = create_bot(args.key_path, args.model, args.base_url)
            raw_html, final_html = generate_dcgen(task["tree"], bot, args.leaf_parallel)
            raw_path.write_text(raw_html, encoding="utf-8")
            final_path.write_text(final_html, encoding="utf-8")
            record["generation_status"] = "ok"
        record["estimated_api_calls"] = (
            task["segmentation"].get("leaf_count", 0) * config["generation"]["num_leaf_generations"] + 1
        )
        record["status"] = "ok"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
    record["elapsed_seconds"] = perf_counter() - started_at
    return record


def render_outputs(output_root, records, replace):
    from playwright.sync_api import sync_playwright
    from utils import get_placeholder, get_tailwind_cdn_script

    render_records = []
    tailwind_script = None
    try:
        tailwind_script = get_tailwind_cdn_script()
    except Exception:
        pass
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        if tailwind_script:
            def serve_tailwind(route):
                if "cdn.tailwindcss.com" in route.request.url:
                    route.fulfill(status=200, content_type="application/javascript", body=tailwind_script)
                else:
                    route.continue_()
            page.route("**/*", serve_tailwind)
        for record in records:
            if record.get("status") != "ok":
                continue
            sample_id = record["sample_id"]
            method_dir = output_root / "outputs" / record["segmenter"]
            for stage, html_path, png_path in (
                ("raw", method_dir / f"{sample_id}.raw.html", method_dir / f"{sample_id}.raw.png"),
                ("final", method_dir / f"{sample_id}.html", method_dir / f"{sample_id}.png"),
            ):
                render_record = {"sample_id": sample_id, "segmenter": record["segmenter"], "stage": stage}
                started_at = perf_counter()
                try:
                    if png_path.is_file() and not replace:
                        render_record["status"] = "reused"
                    else:
                        html = get_placeholder(html_path.read_text(encoding="utf-8", errors="ignore"))
                        page.set_content(html, wait_until="domcontentloaded", timeout=30000)
                        page.screenshot(path=str(png_path), full_page=True, timeout=30000)
                        render_record["status"] = "ok"
                except Exception as exc:
                    render_record["status"] = "failed"
                    render_record["error"] = str(exc)
                render_record["elapsed_seconds"] = perf_counter() - started_at
                render_records.append(render_record)
        browser.close()
    return render_records


def score_outputs(image_dir, output_root, records):
    from scripts.evaluate import CLIPScorer

    scorer = CLIPScorer()
    raw_scores = {BASELINE_NAME: {}, LAYOUTCODER_NAME: {}}
    final_scores = {BASELINE_NAME: {}, LAYOUTCODER_NAME: {}}
    for record in records:
        if record.get("status") != "ok":
            continue
        sample_id = record["sample_id"]
        segmenter = record["segmenter"]
        reference = Image.open(Path(image_dir) / f"{sample_id}.png").convert("RGB")
        method_dir = output_root / "outputs" / segmenter
        raw_image = method_dir / f"{sample_id}.raw.png"
        final_image = method_dir / f"{sample_id}.png"
        if raw_image.is_file():
            raw_scores[segmenter][sample_id] = scorer.score(reference, Image.open(raw_image).convert("RGB"))
        if final_image.is_file():
            final_scores[segmenter][sample_id] = scorer.score(reference, Image.open(final_image).convert("RGB"))
    return raw_scores, final_scores


def mean_scores(score_payload):
    return {
        segmenter: (sum(scores.values()) / len(scores) if scores else None)
        for segmenter, scores in score_payload.items()
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    if not args.key_path:
        raise SystemExit("Provide --key-path or set OPENAI_API_KEY for the full generation experiment")
    if args.generation_workers < 1 or args.segmentation_workers < 1:
        raise SystemExit("worker counts must be positive")
    config = load_config(args.config)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks, records = prepare_segmentation_tasks(args, config, output_root)

    with ThreadPoolExecutor(max_workers=args.generation_workers, thread_name_prefix="dcgen-generation") as executor:
        futures = [executor.submit(run_generation_task, task, args, config, output_root) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            write_json(output_root / "eval" / "generation_records.json", records)
            print(
                f"[{record.get('status')}] {record.get('segmenter')} {record.get('sample_id')} "
                f"{record.get('elapsed_seconds', 0):.1f}s",
                flush=True,
            )

    records.sort(key=lambda record: (record.get("sample_id", ""), record.get("segmenter", "")))
    write_json(output_root / "eval" / "generation_records.json", records)
    if args.render_and_evaluate:
        render_records = render_outputs(output_root, records, args.replace)
        write_json(output_root / "eval" / "render_records.json", render_records)
        raw_scores, final_scores = score_outputs(args.image_dir, output_root, records)
        write_json(output_root / "eval" / "raw_clip_scores.json", raw_scores)
        write_json(output_root / "eval" / "final_clip_scores.json", final_scores)
        summary = {
            "raw_mean_clip": mean_scores(raw_scores),
            "final_mean_clip": mean_scores(final_scores),
            "raw_scores": raw_scores,
            "final_scores": final_scores,
        }
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary["final_mean_clip"], indent=2), flush=True)


if __name__ == "__main__":
    main()
