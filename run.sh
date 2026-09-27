#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${ROOT_DIR}/scripts"
KEY_PATH="${KEY_PATH:-}"
INPUT_DIR="${INPUT_DIR:-}"
DCGEN_OUTPUT_DIR="${DCGEN_OUTPUT_DIR:-}"
DIRECT_OUTPUT_DIR="${DIRECT_OUTPUT_DIR:-}"
EVAL_OUTPUT_PREFIX="${EVAL_OUTPUT_PREFIX:-}"

MODEL_NAME="${MODEL_NAME:-gpt-5.4}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

usage() {
  cat <<'EOF'
用法:
  bash run.sh original-generate
  bash run.sh original-generate-dcgen
  bash run.sh original-generate-direct
  bash run.sh original-screenshot
  bash run.sh original-eval
  bash run.sh original-all

说明:
  original-generate   为 INPUT_DIR 生成两组 HTML
  original-generate-dcgen 只生成 DCGEN_OUTPUT_DIR 的 HTML
  original-generate-direct 只生成 DIRECT_OUTPUT_DIR 的 HTML
  original-screenshot 为两个输出目录里的 HTML 截图
  original-eval       评测输入与两个输出目录
  original-all        依次执行 generate -> screenshot -> eval
可选环境变量:
  MODEL_NAME          默认 gpt-5.4
  OPENAI_API_KEY      API 密钥
  KEY_PATH            仓库外部的 API 密钥文件路径
  OPENAI_BASE_URL     可选的 OpenAI 兼容端点
  INPUT_DIR           用户提供的输入目录
  DCGEN_OUTPUT_DIR    DCGen 输出目录
  DIRECT_OUTPUT_DIR   直接生成输出目录
  EVAL_OUTPUT_PREFIX  评测 JSON 输出前缀
EOF
}

need_key() {
  if [[ -z "${OPENAI_API_KEY:-}" && -z "${KEY_PATH}" ]]; then
    echo "请设置 OPENAI_API_KEY 或 KEY_PATH" >&2
    exit 1
  fi
  if [[ -z "${OPENAI_API_KEY:-}" && ! -f "${KEY_PATH}" ]]; then
    echo "找不到 KEY_PATH 指定的密钥文件" >&2
    exit 1
  fi
}

need_paths() {
  local name
  for name in "$@"; do
    if [[ -z "${!name}" ]]; then
      echo "请设置 ${name}" >&2
      exit 1
    fi
  done
}

setup_env() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  export KEY_PATH
  export INPUT_DIR DCGEN_OUTPUT_DIR DIRECT_OUTPUT_DIR EVAL_OUTPUT_PREFIX
  export OMP_NUM_THREADS
  cd "${SCRIPTS_DIR}"
}

original_generate() {
  need_key
  need_paths INPUT_DIR DCGEN_OUTPUT_DIR DIRECT_OUTPUT_DIR
  setup_env
  python -X utf8 - <<'PY'
import os
from experiments import GPT4, dcgen, single_turn, prompt_direct

bot = GPT4(os.environ.get("OPENAI_API_KEY") or os.environ["KEY_PATH"], model=os.environ.get("MODEL_NAME", "gpt-5.4"))

seg_params = {
    "max_depth": 2,
    "var_thresh": 50,
    "diff_thresh": 45,
    "diff_portion": 0.9,
    "window_size": 50,
}

input_dir = os.environ["INPUT_DIR"]
dcgen_output_dir = os.environ["DCGEN_OUTPUT_DIR"]
direct_output_dir = os.environ["DIRECT_OUTPUT_DIR"]
os.makedirs(dcgen_output_dir, exist_ok=True)
os.makedirs(direct_output_dir, exist_ok=True)

ids = sorted(
    f[:-4]
    for f in os.listdir(input_dir)
    if f.endswith(".png") and f != "placeholder.png"
)

for sample_id in ids:
    out = os.path.join(dcgen_output_dir, f"{sample_id}.html")
    if os.path.exists(out):
        print("DCGen exists, skip:", sample_id)
        continue
    print("DCGen:", sample_id)
    dcgen(bot, os.path.join(input_dir, f"{sample_id}.png"), out, multi_thread=False, seg_params=seg_params)

for sample_id in ids:
    out = os.path.join(direct_output_dir, f"{sample_id}.html")
    if os.path.exists(out):
        print("Direct exists, skip:", sample_id)
        continue
    print("Direct:", sample_id)
    single_turn(prompt_direct, bot, os.path.join(input_dir, f"{sample_id}.png"), out)

for dirname in [dcgen_output_dir, direct_output_dir]:
    placeholder = os.path.join(input_dir, "placeholder.png")
    if os.path.isfile(placeholder):
        import shutil
        shutil.copy2(placeholder, os.path.join(dirname, "placeholder.png"))
PY
}

original_generate_dcgen() {
  need_key
  need_paths INPUT_DIR DCGEN_OUTPUT_DIR
  setup_env
  python -X utf8 - <<'PY'
import os
from experiments import GPT4, dcgen

bot = GPT4(os.environ.get("OPENAI_API_KEY") or os.environ["KEY_PATH"], model=os.environ.get("MODEL_NAME", "gpt-5.4"))

seg_params = {
    "max_depth": 2,
    "var_thresh": 50,
    "diff_thresh": 45,
    "diff_portion": 0.9,
    "window_size": 50,
}

input_dir = os.environ["INPUT_DIR"]
output_dir = os.environ["DCGEN_OUTPUT_DIR"]
os.makedirs(output_dir, exist_ok=True)

ids = sorted(
    f[:-4]
    for f in os.listdir(input_dir)
    if f.endswith(".png") and f != "placeholder.png"
)

for sample_id in ids:
    out = os.path.join(output_dir, f"{sample_id}.html")
    if os.path.exists(out):
        print("DCGen exists, skip:", sample_id)
        continue
    print("DCGen:", sample_id)
    dcgen(bot, os.path.join(input_dir, f"{sample_id}.png"), out, multi_thread=False, seg_params=seg_params)

placeholder = os.path.join(input_dir, "placeholder.png")
if os.path.isfile(placeholder):
    import shutil
    shutil.copy2(placeholder, os.path.join(output_dir, "placeholder.png"))
PY
}

original_generate_direct() {
  need_key
  need_paths INPUT_DIR DIRECT_OUTPUT_DIR
  setup_env
  python -X utf8 - <<'PY'
import os
from experiments import GPT4, single_turn, prompt_direct

bot = GPT4(os.environ.get("OPENAI_API_KEY") or os.environ["KEY_PATH"], model=os.environ.get("MODEL_NAME", "gpt-5.4"))

input_dir = os.environ["INPUT_DIR"]
output_dir = os.environ["DIRECT_OUTPUT_DIR"]
os.makedirs(output_dir, exist_ok=True)

ids = sorted(
    f[:-4]
    for f in os.listdir(input_dir)
    if f.endswith(".png") and f != "placeholder.png"
)

for sample_id in ids:
    out = os.path.join(output_dir, f"{sample_id}.html")
    if os.path.exists(out):
        print("Direct exists, skip:", sample_id)
        continue
    print("Direct:", sample_id)
    single_turn(prompt_direct, bot, os.path.join(input_dir, f"{sample_id}.png"), out)

placeholder = os.path.join(input_dir, "placeholder.png")
if os.path.isfile(placeholder):
    import shutil
    shutil.copy2(placeholder, os.path.join(output_dir, "placeholder.png"))
PY
}

original_screenshot() {
  need_paths DCGEN_OUTPUT_DIR DIRECT_OUTPUT_DIR
  setup_env
  python -X utf8 - <<'PY'
import os
from experiments import take_screenshots_for_dir

take_screenshots_for_dir(os.environ["DCGEN_OUTPUT_DIR"], replace=True)
take_screenshots_for_dir(os.environ["DIRECT_OUTPUT_DIR"], replace=True)
PY
}

original_eval() {
  need_paths INPUT_DIR DCGEN_OUTPUT_DIR DIRECT_OUTPUT_DIR
  setup_env
  python -X utf8 - <<'PY'
import os
from evaluate import clip_experiment, code_sim_experiment

ref = os.environ["INPUT_DIR"]
tests = {
    "dcgen": os.environ["DCGEN_OUTPUT_DIR"],
    "direct": os.environ["DIRECT_OUTPUT_DIR"],
}
exp_name = os.environ.get("EVAL_OUTPUT_PREFIX") or "evaluation"

clip_results = clip_experiment(ref, tests, exp_name)
print("CLIP Results:")
for key in clip_results:
    print(key)
    print(sum(clip_results[key].values()) / len(clip_results[key]))

code_results = code_sim_experiment(ref, tests, exp_name)
print("Code Similarity Results:")
for key in code_results:
    print(key)
    print(sum(code_results[key].values()) / len(code_results[key]))
PY
}

cmd="${1:-}"

export MODEL_NAME

case "${cmd}" in
  original-generate)
    original_generate
    ;;
  original-generate-dcgen)
    original_generate_dcgen
    ;;
  original-generate-direct)
    original_generate_direct
    ;;
  original-screenshot)
    original_screenshot
    ;;
  original-eval)
    original_eval
    ;;
  original-all)
    original_generate
    original_screenshot
    original_eval
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "未知命令: ${cmd}" >&2
    usage
    exit 1
    ;;
esac
