#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from time import monotonic, perf_counter, sleep

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_segmentation_ablation import (
    BROWSER_HEADERS,
    DEFAULT_BASE_URL,
    DIAGNOSTIC_IDS,
    LAYOUTCODER_NAME,
    mean_scores,
    prepare_segmentation_tasks,
    render_outputs,
    run_generation_task,
    score_outputs,
    write_json,
)
from segmentation.layoutcoder_mvp.config import load_config

BASELINE_METHOD = "layoutcoder_mvp_baseline"
CANDIDATE_METHOD = "layoutcoder_structure_flex"
CANDIDATE_PROMPT = """Reproduce the supplied UI crop as one HTML fragment using HTML and Tailwind CSS.
Use placeholder.png for raster images. Match text, colors, spacing, borders, and typography.
The fragment is inserted into an already-sized outer atomic container; do not control page layout.

[STRUCTURE_CONTEXT]
"""


class ApiCapacityGate:
    def __init__(self, cooldown_seconds: float = 120):
        self.cooldown_seconds = cooldown_seconds
        self._resume_at = 0.0
        self._lock = threading.Lock()

    def wait_until_available(self) -> None:
        while True:
            with self._lock:
                remaining = self._resume_at - monotonic()
            if remaining <= 0:
                return
            sleep(min(5, remaining))

    def pause_after_capacity_error(self) -> None:
        with self._lock:
            self._resume_at = max(self._resume_at, monotonic() + self.cooldown_seconds)


def _ask_with_capacity_retry(bot, prompt: str, encoded_crop: object, capacity_gate: ApiCapacityGate) -> tuple[str, int]:
    request_count = 0
    while True:
        capacity_gate.wait_until_available()
        try:
            request_count += 1
            return bot.ask(prompt, encoded_crop), request_count
        except Exception as exc:
            message = str(exc).lower()
            capacity_markers = ("exceed limit", "rate limit", "rate_limit", "too many requests", "429")
            if not any(marker in message for marker in capacity_markers):
                raise
            capacity_gate.pause_after_capacity_error()


def _select_atomic_snippet(
    bot,
    prompt: str,
    encoded_crop: object,
    crop: Image.Image,
    page,
    renderer,
    capacity_gate: ApiCapacityGate,
    initial_candidate_count: int,
) -> tuple[str | None, dict, int]:
    candidate_audits = []
    scored = []
    request_count = 0
    for candidate_index in range(initial_candidate_count):
        candidate, attempts = _ask_with_capacity_retry(bot, prompt, encoded_crop, capacity_gate)
        request_count += attempts
        sanitized, reasons = renderer.sanitize_snippet(candidate or "")
        audit = {"candidate_index": candidate_index, "accepted": not reasons, "sanitation_reasons": reasons}
        if not reasons:
            rendered = _render_html(page, _fragment_document(sanitized, crop.width, crop.height), crop.width, crop.height)
            candidate_mae = _rgb_mae(crop, rendered)
            audit["mae"] = candidate_mae
            scored.append((candidate_mae, candidate_index, sanitized, reasons))
        candidate_audits.append(audit)

    if not scored:
        replacement_index = initial_candidate_count
        replacement, attempts = _ask_with_capacity_retry(bot, prompt, encoded_crop, capacity_gate)
        request_count += attempts
        sanitized, reasons = renderer.sanitize_snippet(replacement or "")
        audit = {"candidate_index": replacement_index, "accepted": not reasons, "sanitation_reasons": reasons}
        if not reasons:
            rendered = _render_html(page, _fragment_document(sanitized, crop.width, crop.height), crop.width, crop.height)
            candidate_mae = _rgb_mae(crop, rendered)
            audit["mae"] = candidate_mae
            scored.append((candidate_mae, replacement_index, sanitized, reasons))
        candidate_audits.append(audit)

    if not scored:
        return None, {"candidates": candidate_audits, "request_count": request_count}, request_count
    selected_mae, selected_index, selected, reasons = min(scored)
    return (
        selected,
        {
            "candidates": candidate_audits,
            "candidate_mae": [audit.get("mae") for audit in candidate_audits],
            "selected_candidate": selected_index,
            "selected_mae": selected_mae,
            "sanitation_reasons": reasons,
            "request_count": request_count,
        },
        request_count,
    )


def _load_matching_candidate_checkpoint(checkpoint_path: Path, identity: dict) -> tuple[dict[int, str], list[dict]]:
    if not checkpoint_path.is_file():
        return {}, []
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("identity") != identity:
        return {}, []
    snippets = {int(atomic_id): snippet for atomic_id, snippet in checkpoint.get("snippets", {}).items()}
    return snippets, checkpoint.get("traces", [])


def _add_shared_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ids", nargs="*", default=DIAGNOSTIC_IDS)
    parser.add_argument("--layoutcoder-root", required=True)


def _add_shared_generation_arguments(parser: argparse.ArgumentParser) -> None:
    _add_shared_input_arguments(parser)
    parser.add_argument("--key-path", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--generation-workers", type=int, default=12)
    parser.add_argument("--segmentation-workers", type=int, default=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LayoutCoder structure and generation pipeline.")
    operations = parser.add_subparsers(dest="operation", required=True)

    baseline_parser = operations.add_parser("baseline", help="Freeze the current LayoutCoder MVP baseline.")
    _add_shared_generation_arguments(baseline_parser)
    baseline_parser.add_argument("--config", default="configs/layoutcoder_segmentation_mvp.yaml")

    structure_parser = operations.add_parser(
        "structure",
        help="Build and validate persisted LayoutCoder structure artifacts without model access.",
    )
    _add_shared_input_arguments(structure_parser)
    structure_parser.add_argument("--config", default="configs/uied_graph_layout_tree.yaml")
    structure_parser.add_argument("--structure-workers", type=int, default=4)

    candidate_parser = operations.add_parser(
        "candidate",
        help="Generate from previously validated structure artifacts.",
    )
    _add_shared_generation_arguments(candidate_parser)
    candidate_parser.add_argument("--config", default="configs/uied_graph_layout_tree.yaml")

    compare_parser = operations.add_parser("compare", help="Compare saved candidate scores with the baseline.")
    compare_parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _prompt_hashes() -> dict[str, str]:
    from scripts.experiments import prompt_dcgen

    return {
        name: hashlib.sha256(prompt_dcgen[name].encode("utf-8")).hexdigest()
        for name in ("prompt_leaf", "prompt_root")
    }


def _baseline_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        image_dir=args.image_dir,
        ids=args.ids,
        layoutcoder_root=args.layoutcoder_root,
        segmentation_workers=args.segmentation_workers,
        key_path=args.key_path,
        model=args.model,
        base_url=args.base_url,
        generation_workers=args.generation_workers,
        leaf_parallel=False,
        replace=False,
    )


def run_baseline(args: argparse.Namespace) -> dict:
    if args.generation_workers < 1 or args.segmentation_workers < 1:
        raise ValueError("worker counts must be positive")
    config = load_config(args.config)
    baseline_root = Path(args.output_dir) / "baseline_results"
    baseline_root.mkdir(parents=True, exist_ok=True)
    runtime_args = _baseline_runtime_args(args)
    tasks, records = prepare_segmentation_tasks(runtime_args, config, baseline_root)
    baseline_tasks = [task for task in tasks if task["segmenter"] == LAYOUTCODER_NAME]
    with ThreadPoolExecutor(max_workers=args.generation_workers, thread_name_prefix="layoutcoder-baseline") as executor:
        futures = [executor.submit(run_generation_task, task, runtime_args, config, baseline_root) for task in baseline_tasks]
        for future in as_completed(futures):
            record = future.result()
            record["method"] = BASELINE_METHOD
            records.append(record)
            write_json(baseline_root / "generation_records.json", records)
    records.sort(key=lambda record: str(record.get("sample_id", "")))
    write_json(baseline_root / "generation_records.json", records)
    render_records = render_outputs(baseline_root, records, replace=False)
    write_json(baseline_root / "render_records.json", render_records)
    raw_scores, final_scores = score_outputs(args.image_dir, baseline_root, records)
    write_json(baseline_root / "raw_clip_scores.json", raw_scores)
    write_json(baseline_root / "final_clip_scores.json", final_scores)
    successful_records = [
        record for record in records if record.get("segmenter") == LAYOUTCODER_NAME and record.get("status") == "ok"
    ]
    summary = {
        "method": BASELINE_METHOD,
        "source_segmenter": LAYOUTCODER_NAME,
        "sample_ids": list(args.ids),
        "sample_count": len(args.ids),
        "generation_success_count": len(successful_records),
        "render_success_count": sum(record.get("status") in {"ok", "reused"} for record in render_records),
        "raw_clip_count": len(raw_scores[LAYOUTCODER_NAME]),
        "final_clip_count": len(final_scores[LAYOUTCODER_NAME]),
        "raw_mean_clip": mean_scores(raw_scores)[LAYOUTCODER_NAME],
        "final_mean_clip": mean_scores(final_scores)[LAYOUTCODER_NAME],
        "estimated_api_calls": sum(record.get("estimated_api_calls", 0) for record in successful_records),
        "effective_config": config,
        "prompt_hashes": _prompt_hashes(),
        "viewport": [1920, 1080],
        "browser_headers": sorted(BROWSER_HEADERS),
    }
    write_json(baseline_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _candidate_prompt_hash() -> str:
    return hashlib.sha256(CANDIDATE_PROMPT.encode("utf-8")).hexdigest()


def _render_html(page, html: str, width: int, height: int) -> Image.Image:
    from utils import get_placeholder

    page.set_viewport_size({"width": max(1, width), "height": max(1, height)})
    page.set_content(get_placeholder(html), wait_until="domcontentloaded", timeout=30000)
    screenshot = page.screenshot(full_page=True, animations="disabled", timeout=30000)
    return Image.open(io.BytesIO(screenshot)).convert("RGB").resize((max(1, width), max(1, height)))


def _fragment_document(fragment: str, width: int, height: int) -> str:
    return (
        "<!DOCTYPE html><html><head><meta charset=\"UTF-8\">"
        "<script src=\"https://cdn.tailwindcss.com\"></script>"
        "<style>html,body{margin:0;width:100%;height:100%;overflow:hidden}*{box-sizing:border-box}</style>"
        f"</head><body><div style=\"width:{width}px;height:{height}px;overflow:hidden\">{fragment}</div></body></html>"
    )


def _rgb_mae(reference: Image.Image, rendered: Image.Image) -> float:
    reference_values = np.asarray(reference.convert("RGB"), dtype=np.int16)
    rendered_values = np.asarray(rendered.convert("RGB").resize(reference.size), dtype=np.int16)
    return float(np.mean(np.abs(reference_values - rendered_values)))


def _sample_root(output_dir: str, sample_id: str) -> Path:
    return Path(output_dir) / "candidate_results" / "samples" / sample_id


def _load_passing_structure(output_dir: str, sample_id: str):
    from layout_pipeline.schema import ElementNode, LayoutNode

    sample_root = _sample_root(output_dir, sample_id)
    validation_path = sample_root / "validation.json"
    structure_path = sample_root / "structure.json"
    elements_path = sample_root / "merged_uied.json"
    if not validation_path.is_file() or not structure_path.is_file() or not elements_path.is_file():
        raise RuntimeError(f"sample {sample_id} has no persisted structure validation")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("passed") is not True:
        raise RuntimeError(f"sample {sample_id} structure validation failed")
    structure = LayoutNode.from_dict(json.loads(structure_path.read_text(encoding="utf-8")))
    element_payloads = json.loads(elements_path.read_text(encoding="utf-8"))["elements"]
    elements = [
        ElementNode(
            int(payload["id"]),
            tuple(payload["bbox"]),
            payload["class"],
            payload.get("text"),
            payload["segmentation_role"],
        )
        for payload in element_payloads
    ]
    return structure, elements, validation


def _tailwind_page(playwright):
    from utils import get_tailwind_cdn_script

    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    tailwind_script = get_tailwind_cdn_script()

    def serve_tailwind(route):
        if route.request.url == "https://cdn.tailwindcss.com/":
            route.fulfill(status=200, content_type="application/javascript", body=tailwind_script)
        else:
            route.continue_()

    page.route("**/*", serve_tailwind)
    return browser, page


def _run_structure_sample(sample_id: str, args: argparse.Namespace, config: dict) -> dict:
    from playwright.sync_api import sync_playwright

    from layout_pipeline.diagnostics import LayoutDiagnosticsWriter
    from layout_pipeline.pipeline import LayoutCoderStructurePipeline
    from layout_pipeline.semantic_renderer import SemanticLayoutRenderer

    started_at = perf_counter()
    image_path = Path(args.image_dir) / f"{sample_id}.png"
    sample_root = _sample_root(args.output_dir, sample_id)
    record = {"sample_id": sample_id, "operation": "structure"}
    if not image_path.is_file():
        return {**record, "status": "missing_image", "error": str(image_path)}
    try:
        pipeline = LayoutCoderStructurePipeline(config, args.layoutcoder_root)
        uied_root = sample_root / "uied"
        merged_path = uied_root / "merge" / f"{sample_id}.json"
        if merged_path.is_file():
            artifacts = pipeline.backend.load_artifacts(str(image_path), str(uied_root))
            analysis = pipeline.analyze_artifacts(str(image_path), artifacts)
        else:
            analysis = pipeline.analyze(str(image_path), str(uied_root))
        LayoutDiagnosticsWriter().write(str(image_path), str(sample_root), analysis)
        skeleton_html = SemanticLayoutRenderer().render_placeholders(analysis.structure)
        (sample_root / "skeleton.html").write_text(skeleton_html, encoding="utf-8")
        with Image.open(image_path) as opened:
            width, height = opened.size
        with sync_playwright() as playwright:
            browser, page = _tailwind_page(playwright)
            _render_html(page, skeleton_html, width, height).save(sample_root / "skeleton.png")
            browser.close()
        record.update(
            {
                "status": "ok" if analysis.validation.passed else "failed",
                "validation": analysis.validation.to_dict(),
                "second_pass_used": analysis.second_pass_used,
                "elapsed_seconds": perf_counter() - started_at,
            }
        )
    except Exception as exc:
        record.update({"status": "failed", "error": str(exc), "elapsed_seconds": perf_counter() - started_at})
    write_json(sample_root / "structure_summary.json", record)
    return record


def run_structure(args: argparse.Namespace) -> dict:
    if args.structure_workers < 1:
        raise ValueError("structure worker count must be positive")
    config = load_config(args.config)
    records = []
    with ThreadPoolExecutor(max_workers=args.structure_workers, thread_name_prefix="layoutcoder-structure") as executor:
        futures = [executor.submit(_run_structure_sample, str(sample_id), args, config) for sample_id in args.ids]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: int(record["sample_id"]))
    summary = {
        "operation": "structure",
        "sample_ids": [str(sample_id) for sample_id in args.ids],
        "passed": all(record.get("status") == "ok" for record in records),
        "records": records,
    }
    write_json(Path(args.output_dir) / "candidate_results" / "structure_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _generate_candidate_sample(
    sample_id: str,
    args: argparse.Namespace,
    config: dict,
    capacity_gate: ApiCapacityGate,
) -> dict:
    from playwright.sync_api import sync_playwright

    from layout_pipeline.diagnostics import write_json as write_diagnostic_json
    from layout_pipeline.semantic_renderer import SemanticLayoutRenderer, leaf_nodes, parent_by_leaf_id
    from layout_pipeline.structure_context import LeafStructureContextBuilder
    from scripts.run_segmentation_ablation import create_bot
    from utils import encode_image

    started_at = perf_counter()
    candidate_root = Path(args.output_dir) / "candidate_results"
    sample_root = _sample_root(args.output_dir, sample_id)
    method_root = candidate_root / "outputs" / CANDIDATE_METHOD
    method_root.mkdir(parents=True, exist_ok=True)
    image_path = Path(args.image_dir) / f"{sample_id}.png"
    record = {"sample_id": sample_id, "method": CANDIDATE_METHOD}
    actual_api_calls = 0
    try:
        structure, elements, validation = _load_passing_structure(args.output_dir, sample_id)
        renderer = SemanticLayoutRenderer()
        atomics = leaf_nodes(structure)
        parents = parent_by_leaf_id(structure)
        context_builder = LeafStructureContextBuilder(structure, elements)
        with Image.open(image_path) as opened:
            reference = opened.convert("RGB")
        bot = create_bot(args.key_path, args.model, args.base_url)
        checkpoint_path = sample_root / "atomic_generation_checkpoint.json"
        checkpoint_identity = {
            "image_hash": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "structure_hash": hashlib.sha256(
                json.dumps(structure.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "prompt_hash": _candidate_prompt_hash(),
            "model": args.model,
            "num_leaf_generations": config["generation"]["num_leaf_generations"],
        }
        snippets, traces = _load_matching_candidate_checkpoint(checkpoint_path, checkpoint_identity)
        actual_api_calls = sum(int(trace.get("request_count", 0)) for trace in traces)
        with sync_playwright() as playwright:
            browser, page = _tailwind_page(playwright)
            for atomic in atomics:
                if atomic.id in snippets:
                    continue
                crop = reference.crop(atomic.bbox)
                crop_path = sample_root / "atomic_crops" / f"{atomic.id}.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path)
                context = context_builder.build(atomic, parents.get(atomic.id))
                prompt = CANDIDATE_PROMPT.replace("[STRUCTURE_CONTEXT]", context)
                encoded_crop = encode_image(crop)
                selected, trace, request_count = _select_atomic_snippet(
                    bot,
                    prompt,
                    encoded_crop,
                    crop,
                    page,
                    renderer,
                    capacity_gate,
                    config["generation"]["num_leaf_generations"],
                )
                actual_api_calls += request_count
                trace.update({"atomic_id": atomic.id, "context": context})
                traces.append(trace)
                if selected is None:
                    write_diagnostic_json(
                        checkpoint_path,
                        {
                            "identity": checkpoint_identity,
                            "snippets": {str(atomic_id): snippet for atomic_id, snippet in snippets.items()},
                            "traces": traces,
                        },
                    )
                    raise RuntimeError(f"all generated candidates rejected for atomic {atomic.id}")
                snippets[atomic.id] = selected
                write_diagnostic_json(
                    checkpoint_path,
                    {
                        "identity": checkpoint_identity,
                        "snippets": {str(atomic_id): snippet for atomic_id, snippet in snippets.items()},
                        "traces": traces,
                    },
                )
            final = renderer.render_final(structure, snippets)
            final_html_path = method_root / f"{sample_id}.html"
            final_html_path.write_text(final.html, encoding="utf-8")
            _render_html(page, final.html, reference.width, reference.height).save(method_root / f"{sample_id}.png")
            browser.close()
        write_diagnostic_json(sample_root / "atomic_generation_trace.json", traces)
        write_diagnostic_json(sample_root / "atomic_snippet_audit.json", [audit.to_dict() for audit in final.audits])
        record.update(
            {
                "status": "ok",
                "atomic_count": len(atomics),
                "thin_text_atomic_ratio": validation["thin_text_atomic_ratio"],
                "structure_metrics": {
                    name: value
                    for name, value in validation.items()
                    if name not in {"passed", "errors"}
                },
                "estimated_api_calls": len(atomics) * config["generation"]["num_leaf_generations"],
                "actual_api_calls": actual_api_calls,
                "elapsed_seconds": perf_counter() - started_at,
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "error": str(exc),
                "actual_api_calls": actual_api_calls,
                "elapsed_seconds": perf_counter() - started_at,
            }
        )
    write_json(sample_root / "sample_summary.json", record)
    return record


def run_candidate(args: argparse.Namespace) -> dict:
    if args.generation_workers < 1:
        raise ValueError("generation worker count must be positive")
    config = load_config(args.config)
    for sample_id in args.ids:
        _load_passing_structure(args.output_dir, str(sample_id))
    candidate_root = Path(args.output_dir) / "candidate_results"
    candidate_root.mkdir(parents=True, exist_ok=True)
    records = []
    capacity_gate = ApiCapacityGate()
    with ThreadPoolExecutor(max_workers=args.generation_workers, thread_name_prefix="layoutcoder-candidate") as executor:
        future_by_id = {
            executor.submit(_generate_candidate_sample, str(sample_id), args, config, capacity_gate): str(sample_id)
            for sample_id in args.ids
        }
        for future in as_completed(future_by_id):
            records.append(future.result())
            write_json(candidate_root / "generation_records.json", sorted(records, key=lambda item: item["sample_id"]))
    records.sort(key=lambda item: item["sample_id"])
    write_json(candidate_root / "generation_records.json", records)
    from scripts.evaluate import CLIPScorer

    scorer = CLIPScorer()
    scores = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        sample_id = record["sample_id"]
        with Image.open(Path(args.image_dir) / f"{sample_id}.png") as opened:
            reference = opened.convert("RGB")
        with Image.open(candidate_root / "outputs" / CANDIDATE_METHOD / f"{sample_id}.png") as opened:
            candidate = opened.convert("RGB")
        scores[sample_id] = scorer.score(reference, candidate)
        record["clip"] = scores[sample_id]
        write_json(_sample_root(args.output_dir, sample_id) / "sample_summary.json", record)
    score_payload = {CANDIDATE_METHOD: scores}
    write_json(candidate_root / "final_clip_scores.json", score_payload)
    summary = {
        "method": CANDIDATE_METHOD,
        "sample_ids": [str(sample_id) for sample_id in args.ids],
        "generation_success_count": sum(record.get("status") == "ok" for record in records),
        "render_success_count": sum(record.get("status") == "ok" for record in records),
        "final_clip_count": len(scores),
        "final_mean_clip": mean(scores.values()) if scores else None,
        "estimated_api_calls": sum(record.get("estimated_api_calls", 0) for record in records),
        "actual_api_calls": sum(record.get("actual_api_calls", 0) for record in records),
        "per_sample": [
            {
                "sample_id": record["sample_id"],
                "atomic_count": record.get("atomic_count"),
                "thin_text_atomic_ratio": record.get("thin_text_atomic_ratio"),
                "estimated_api_calls": record.get("estimated_api_calls", 0),
                "actual_api_calls": record.get("actual_api_calls", 0),
                "structure_metrics": record.get("structure_metrics", {}),
                "clip": record.get("clip"),
                "status": record.get("status"),
            }
            for record in records
        ],
        "effective_config": config,
        "prompt_hashes": {"candidate_atomic": _candidate_prompt_hash()},
        "viewport": [1920, 1080],
    }
    write_json(candidate_root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def compare_results(output_dir: str) -> dict:
    experiment_root = Path(output_dir)
    baseline_payload = json.loads(
        (experiment_root / "baseline_results" / "final_clip_scores.json").read_text(encoding="utf-8")
    )
    candidate_payload = json.loads((experiment_root / "candidate_results" / "final_clip_scores.json").read_text(encoding="utf-8"))
    baseline_scores = baseline_payload[LAYOUTCODER_NAME]
    candidate_scores = candidate_payload[CANDIDATE_METHOD]
    sample_ids = sorted(set(baseline_scores) & set(candidate_scores), key=lambda value: int(value))
    per_sample = [
        {
            "sample_id": sample_id,
            "baseline": baseline_scores[sample_id],
            "candidate": candidate_scores[sample_id],
            "delta": candidate_scores[sample_id] - baseline_scores[sample_id],
        }
        for sample_id in sample_ids
    ]
    deltas = [record["delta"] for record in per_sample]
    generation_records_path = experiment_root / "candidate_results" / "generation_records.json"
    generation_records = json.loads(generation_records_path.read_text(encoding="utf-8")) if generation_records_path.is_file() else []
    candidate_mean = mean(candidate_scores.values()) if candidate_scores else None
    actual_api_calls = sum(record.get("actual_api_calls", 0) for record in generation_records)
    delta_by_sample = {record["sample_id"]: record["delta"] for record in per_sample}
    summary = {
        "sample_count": len(sample_ids),
        "baseline_mean": mean(baseline_scores[sample_id] for sample_id in sample_ids) if sample_ids else None,
        "candidate_mean": candidate_mean,
        "mean_delta": mean(deltas) if deltas else None,
        "median_delta": median(deltas) if deltas else None,
        "wins": sum(delta > 0 for delta in deltas),
        "regressions_below_minus_0_02": sum(delta < -0.02 for delta in deltas),
        "render_failures": sum(record.get("status") != "ok" for record in generation_records),
        "actual_api_calls": actual_api_calls,
        "per_sample": per_sample,
    }
    summary["success_gate"] = {
        "candidate_mean": summary["candidate_mean"] is not None and summary["candidate_mean"] >= 0.948,
        "median_delta": summary["median_delta"] is not None and summary["median_delta"] > 0,
        "wins": summary["wins"] >= 8,
        "regressions": summary["regressions_below_minus_0_02"] <= 1,
        "sample_22": delta_by_sample.get("22", float("-inf")) >= 0.02,
        "sample_91": delta_by_sample.get("91", float("-inf")) >= 0.03,
        "sample_12": delta_by_sample.get("12", float("-inf")) >= -0.01,
        "sample_13": delta_by_sample.get("13", float("-inf")) >= -0.01,
        "actual_api_calls": actual_api_calls <= 450,
        "render_failures": summary["render_failures"] == 0,
    }
    summary["passed"] = len(sample_ids) == 12 and all(summary["success_gate"].values())
    write_json(experiment_root / "comparison.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.operation in {"baseline", "candidate"} and not args.key_path:
        raise SystemExit("Provide --key-path or set OPENAI_API_KEY for generation")
    if args.operation == "baseline":
        run_baseline(args)
    elif args.operation == "structure":
        run_structure(args)
    elif args.operation == "candidate":
        run_candidate(args)
    else:
        print(json.dumps(compare_results(args.output_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()
