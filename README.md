# CREATE: Screenshot-to-HTML Generation via Region-Aware Tree Search

**CREATE** stands for s**C**reenshot-to-HTML gene**R**ation via r**E**gion-**A**ware **T**ree s**E**arch.

CREATE converts a webpage screenshot into HTML by first recovering a validated layout tree and then generating the visual content of its atomic regions. Instead of using a fixed grid or a fixed segmentation depth, CREATE adapts the granularity of the tree to the visual and semantic structure of each page.

The repository contains the CREATE implementation, its LayoutCoder/UIED integration, experiment runners, and tests. Generated experiment outputs, credentials, datasets, and the external LayoutCoder checkout are not included.

## Method overview

```text
Screenshot
  -> LayoutCoder/UIED element detection and OCR
  -> separator detection and layout grouping
  -> semantic region profiling
  -> region-aware split proposal and tree search
  -> atomic ownership, merging, and structural validation
  -> per-atomic HTML generation and candidate selection
  -> tree-guided HTML composition and rendering
```

The structure stage combines several signals when deciding whether and where to split a region:

- visual separators and whitespace;
- UI element ownership and repeated layout groups;
- region types such as hero, navigation, text section, card grid, two-column layout, list, and footer;
- penalties for fragmenting text or large visual components;
- structural quality gates and estimated generation cost.

Candidate trees are ranked with a bounded search. CREATE selects the highest-ranked tree that passes structural validation, assigns every detected element to an atomic region, replaces layout-only empty regions with spacers, and merges text regions separated only by weak boundaries. Each remaining atomic region is generated independently with its local crop and structural context before the snippets are assembled through the recovered tree.

## Repository layout

```text
configs/
  uied_graph_layout_tree.yaml       # CREATE structure, search, and generation configuration
layout_pipeline/                    # Region-aware tree construction and validation
segmentation/layoutcoder_mvp/       # LayoutCoder/UIED integration and baseline adapter
scripts/
  run_uied_graph_layout_pipeline.py # Main structure, generation, and comparison runner
  run_segmentation_ablation.py      # LayoutCoder segmentation baseline runner
tests/layout_pipeline/              # CREATE unit and runner tests
requirements-layoutcoder-mvp.txt    # LayoutCoder/UIED-side Python dependencies
requirements.txt                    # Original DCGen and utility dependencies
```

## Requirements

- Python with a PaddlePaddle/PaddleOCR combination supported by the target platform;
- Chromium installed through Playwright;
- an external LayoutCoder checkout at the revision pinned by the CREATE configuration;
- an OpenAI-compatible vision model only for the generation stages.

Install the repository dependencies in an isolated environment:

```bash
pip install -r requirements.txt
pip install -r requirements-layoutcoder-mvp.txt
pip install playwright pyyaml
playwright install chromium
```

PaddlePaddle installation is platform-specific. Install a build compatible with the machine before running UIED/PaddleOCR. The dependency files describe the packages used by this repository, but they do not replace the platform-specific PaddlePaddle installation instructions.

CREATE uses the external [LayoutCoder](https://github.com/ay7u1009/LayoutCoder) repository for UI element detection. Check out the exact revision recorded in `configs/uied_graph_layout_tree.yaml`:

```bash
git clone https://github.com/ay7u1009/LayoutCoder.git /path/to/LayoutCoder
git -C /path/to/LayoutCoder checkout bf5b0032923ea68a0aff9f98fa9cd544d8cd9ee8
```

The runner verifies the LayoutCoder revision before processing a screenshot and stops on a mismatch.

## Input convention

Commands accept a directory of PNG screenshots and a list of numeric sample IDs. Each ID resolves to one input file:

```text
/path/to/screenshots/
  0.png
  1.png
  2.png
```

Pass the corresponding IDs as `--ids 0 1 2`. If `--ids` is omitted, the runner uses its built-in diagnostic cohort rather than discovering every image in the directory.

## Quick start

### 1. Build and validate the region-aware tree

The `structure` operation performs UI detection, tree search, structural validation, and diagnostic rendering. It does **not** construct a model client or call a model API.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_uied_graph_layout_pipeline.py structure \
  --image-dir /path/to/screenshots \
  --output-dir /path/to/create-run \
  --ids 0 1 2 \
  --layoutcoder-root /path/to/LayoutCoder \
  --config configs/uied_graph_layout_tree.yaml \
  --structure-workers 1
```

Before continuing, inspect the command summary or each sample's `validation.json`. A sample is eligible for generation only when `passed` is `true`. The runner enforces this requirement and rejects missing or failed structure artifacts before model access.

After a successful structure run, important artifacts are written under:

```text
/path/to/create-run/candidate_results/samples/<sample-id>/
  merged_uied.json
  structure.json
  validation.json
  structure_summary.json
  skeleton.html
  skeleton.png
  atomic_ownership.json
  beam_states.json
  region_profiles.json
  tree_score_breakdown.json
  structure_overlay.png
  uied/
```

### 2. Generate and compose atomic regions

The `candidate` operation calls the configured vision model and may incur API charges. Supply credentials at runtime; do not store them in the repository.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"

python scripts/run_uied_graph_layout_pipeline.py candidate \
  --image-dir /path/to/screenshots \
  --output-dir /path/to/create-run \
  --ids 0 1 2 \
  --layoutcoder-root /path/to/LayoutCoder \
  --config configs/uied_graph_layout_tree.yaml \
  --model gpt-5.4 \
  --generation-workers 1
```

Use `OPENAI_BASE_URL` or `--base-url` for an OpenAI-compatible endpoint. The runner also accepts `--key-path` when the credential is stored in a file outside the repository.

For every atomic region, CREATE saves its crop, generation trace, accepted snippet, and checkpoint state. Checkpoints are reused only when the image, recovered structure, prompt, model, and number of generations match. Successful final outputs are written under:

```text
/path/to/create-run/candidate_results/
  outputs/layoutcoder_structure_flex/<sample-id>.html
  outputs/layoutcoder_structure_flex/<sample-id>.png
  generation_records.json
  final_clip_scores.json
  summary.json
```

`layoutcoder_structure_flex` is the internal artifact identifier currently used by the runner; the public method name is CREATE.

## Baseline and comparison

The optional `baseline` operation runs the LayoutCoder MVP segmentation baseline and calls the configured model API:

```bash
python scripts/run_uied_graph_layout_pipeline.py baseline \
  --image-dir /path/to/screenshots \
  --output-dir /path/to/create-run \
  --ids 0 1 2 \
  --layoutcoder-root /path/to/LayoutCoder \
  --config configs/layoutcoder_segmentation_mvp.yaml \
  --model gpt-5.4 \
  --generation-workers 1 \
  --segmentation-workers 1
```

After both baseline and CREATE score files exist under the same output directory, compare their common successfully scored samples:

```bash
python scripts/run_uied_graph_layout_pipeline.py compare \
  --output-dir /path/to/create-run
```

The comparison writes `comparison.json`. Missing or failed generations are not assigned synthetic zero scores; the comparison is computed over the intersection of available baseline and CREATE scores.

## Configuration

The primary configuration is `configs/uied_graph_layout_tree.yaml`. It records:

- mask and separator thresholds;
- region-specific split policies;
- tree-search limits and scoring weights;
- structural quality gates;
- atomic ownership and rendering constraints;
- generation settings;
- the required LayoutCoder revision.

For reproducible experiments, preserve the configuration together with the input IDs, input image hashes, model identifier, prompt version, and output directory. A configured model name or endpoint does not by itself confirm that the provider makes that model available.

## Tests

Run the repository test suite from the repository root:

```bash
python -m unittest discover -s tests -t . -p "test_*.py"
```

The suite covers the region classifier, split policy, tree search and scoring, atomic ownership and merging, structure validation, renderer, checkpoint matching, and runner boundaries. It does not execute the external LayoutCoder pipeline against a real screenshot and does not call a model API.

## Legacy utilities

The repository still includes the original DCGen utilities in `utils.py`, `scripts/experiments.py`, the webpage capture helpers in `scripts/single_file.py`, and the local interface under `Tool/`. They are retained as supporting and baseline code; the main CREATE workflow is `scripts/run_uied_graph_layout_pipeline.py`.

To start the legacy local interface:

```bash
cd Tool
python app.py
```

Then open <http://127.0.0.1:8080> in a local browser.

## Acknowledgements

CREATE builds on the UI element detection and OCR integration provided by LayoutCoder, UIED, and PaddleOCR, and retains DCGen components for baseline generation and evaluation.
