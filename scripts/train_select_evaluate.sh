#!/usr/bin/env bash
# Complete VM -> CVM -> StraightPCF training, selection and local evaluation.
# The repository has no SVM class; the requested SVM stage is CVM
# (CoupledVelocityModule), the documented second stage of StraightPCF.

set -Eeuo pipefail
IFS=$'\n\t'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STAGES=(
  prepare-split
  prepare-cache
  train-vm
  select-vm
  train-cvm
  select-cvm
  train-straightpcf
  select-straightpcf
  evaluate
)

FROM_STAGE="prepare-split"
TO_STAGE="evaluate"
DRY_RUN=0
CHECK_ONLY=0
RESUME=0
RESUME_SELECTION=0
OVERWRITE_SPLIT=0
OVERWRITE_CACHE=0
SEED=123
USE_CUDA=1
WORKERS=16
TEST_RATIO=0.02
MIN_TEST_PER_CATEGORY=2
TRAIN_NUM_POINTS=200000
TEST_NUM_POINTS=50000
PREFILTER_TOP_K=10
PREFILTER_LAST_N=0
PREFILTER_CD_POINTS=8192
PREFILTER_CD_LIMIT=20
FUSION_MODE=max
EVAL_LIMIT=""
RESULT_FILE="pipeline_results/local_test_results.txt"

if [[ -n "${CONDA_PREFIX:-}" && "$(basename "$CONDA_PREFIX")" == "jittor2A" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -x /root/miniconda3/envs/jittor2A/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/envs/jittor2A/bin/python
else
  PYTHON_BIN=python
fi

usage() {
  cat <<'USAGE'
Usage: bash scripts/train_select_evaluate.sh [options]

Stages:
  prepare-split -> prepare-cache -> train-vm -> select-vm ->
  train-cvm -> select-cvm -> train-straightpcf ->
  select-straightpcf -> evaluate

Options:
  --check-only                 Check code/config/dependencies/data readiness.
  --dry-run                    Print commands without executing them.
  --from-stage NAME            Start at NAME (default: prepare-split).
  --to-stage NAME              Stop after NAME (default: evaluate).
  --resume                     Skip stages with their completion artifact.
  --resume-selection           Pass --resume to select_best_checkpoint.py.
  --overwrite-split            Rebuild local_train/local_test symlink trees.
  --overwrite-cache            Regenerate point-cloud caches.
  --python PATH                Python interpreter (defaults to jittor2A).
  --seed N                     Random seed (default: 123).
  --workers N                  Cache workers (default: 16).
  --use-cuda 0|1               Jittor CUDA flag (default: 1).
  --prefilter-top-k N          Full-score candidates (default: 10).
  --prefilter-last-n N         Quick-CD epochs; 0 means all (default: 0).
  --prefilter-cd-points N      Quick-CD point count (default: 8192).
  --prefilter-cd-limit N       Quick-CD sample count (default: 20).
  --eval-limit N               Evaluate only the first N local-test samples.
  --fusion-mode max|mix        Final inference fusion (default: max).
  --result-file PATH           Final local-test result file.
  -h, --help                   Show this help.

Examples:
  bash scripts/train_select_evaluate.sh --check-only
  bash scripts/train_select_evaluate.sh --dry-run
  bash scripts/train_select_evaluate.sh
  bash scripts/train_select_evaluate.sh \
    --from-stage select-vm --resume-selection
USAGE
}

need_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) CHECK_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --resume) RESUME=1; shift ;;
    --resume-selection) RESUME_SELECTION=1; shift ;;
    --overwrite-split) OVERWRITE_SPLIT=1; shift ;;
    --overwrite-cache) OVERWRITE_CACHE=1; shift ;;
    --from-stage) need_value "$@"; FROM_STAGE="$2"; shift 2 ;;
    --to-stage) need_value "$@"; TO_STAGE="$2"; shift 2 ;;
    --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --seed) need_value "$@"; SEED="$2"; shift 2 ;;
    --workers) need_value "$@"; WORKERS="$2"; shift 2 ;;
    --use-cuda) need_value "$@"; USE_CUDA="$2"; shift 2 ;;
    --prefilter-top-k) need_value "$@"; PREFILTER_TOP_K="$2"; shift 2 ;;
    --prefilter-last-n) need_value "$@"; PREFILTER_LAST_N="$2"; shift 2 ;;
    --prefilter-cd-points) need_value "$@"; PREFILTER_CD_POINTS="$2"; shift 2 ;;
    --prefilter-cd-limit) need_value "$@"; PREFILTER_CD_LIMIT="$2"; shift 2 ;;
    --eval-limit) need_value "$@"; EVAL_LIMIT="$2"; shift 2 ;;
    --fusion-mode) need_value "$@"; FUSION_MODE="$2"; shift 2 ;;
    --result-file) need_value "$@"; RESULT_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

stage_index() {
  local wanted="$1"
  local index
  for index in "${!STAGES[@]}"; do
    if [[ "${STAGES[$index]}" == "$wanted" ]]; then
      echo "$index"
      return 0
    fi
  done
  echo "Unknown stage: $wanted" >&2
  return 1
}

FROM_INDEX="$(stage_index "$FROM_STAGE")"
TO_INDEX="$(stage_index "$TO_STAGE")"
if (( FROM_INDEX > TO_INDEX )); then
  echo "--from-stage must not come after --to-stage" >&2
  exit 2
fi
if [[ "$USE_CUDA" != 0 && "$USE_CUDA" != 1 ]]; then
  echo "--use-cuda must be 0 or 1" >&2
  exit 2
fi
if [[ "$FUSION_MODE" != max && "$FUSION_MODE" != mix ]]; then
  echo "--fusion-mode must be max or mix" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REQUIRED_FILES=(
  run.py
  select_best_checkpoint.py
  src/config_override.py
  scripts/create_local_holdout.py
  scripts/precompute_clean_points.py
  scripts/generate_local_test_benchmark.py
  scripts/evaluate_local_test_models.py
  configs/data/train_cached.yaml
  configs/task/train_vm_cached.yaml
  configs/task/train_cvm_cached.yaml
  configs/task/train_straightpcf_cached.yaml
  configs/model/cvm.yaml
  configs/model/straightpcf.yaml
)

preflight() {
  local failed=0
  local path
  echo "=== Preflight ==="
  echo "Python: $PYTHON_BIN"
  for path in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$path" ]]; then
      echo "ERROR missing: $path" >&2
      failed=1
    fi
  done
  if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR Python not found: $PYTHON_BIN" >&2
    failed=1
  elif ! "$PYTHON_BIN" -c 'import jittor,numpy,scipy,trimesh,omegaconf,point_cloud_utils' >/dev/null 2>&1; then
    echo "ERROR missing Python dependencies; run: python -m pip install -r requirements.txt" >&2
    failed=1
  else
    echo "Dependencies: OK"
  fi
  if ! grep -Fq 'dataset_train_pcd_disk/local_train' configs/data/train_cached.yaml; then
    echo "ERROR train_cached.yaml is not wired to the generated local_train cache" >&2
    failed=1
  fi
  if ! grep -Fq 'checkpoint_selection_cached/best_checkpoint.pkl' configs/model/cvm.yaml; then
    echo "ERROR cvm.yaml init_velocity_ckpt does not match VM selection output" >&2
    failed=1
  fi
  if ! grep -Fq 'checkpoint_selection_cvm/best_checkpoint.pkl' configs/model/straightpcf.yaml; then
    echo "ERROR straightpcf.yaml init_cvm_ckpt does not match CVM selection output" >&2
    failed=1
  fi
  if grep -Eq 'lr_decay|lr_min_ratio|math\.cos' src/system/spec.py; then
    echo "ERROR learning-rate scheduler logic is still present in src/system/spec.py" >&2
    failed=1
  else
    echo "Fixed learning rate: OK"
  fi
  local mesh_count=0
  local noisy_count=0
  if [[ -d dataset_train/shapenet ]]; then
    mesh_count="$(find dataset_train/shapenet -type f -path '*/models/model_normalized.obj' | wc -l)"
  fi
  if [[ -d dataset_test_noisy ]]; then
    noisy_count="$(find dataset_test_noisy -type f -name noisy.npy | wc -l)"
  fi
  echo "Source meshes currently present: $mesh_count"
  echo "Test noisy clouds currently present: $noisy_count"
  [[ -f dataset_train/local_train/datalist.txt ]] && echo "Local split: ready" || echo "Local split: not created"
  [[ -f dataset_train_pcd_disk/precompute_manifest.json ]] && echo "Cache: ready" || echo "Cache: not created"
  return "$failed"
}

print_command() {
  printf '$'
  printf ' %q' "$@"
  printf '\n'
}

CURRENT_STAGE=""
trap 'status=$?; echo "FAILED at stage ${CURRENT_STAGE:-preflight} (exit $status)" >&2; exit "$status"' ERR

run_stage() {
  local name="$1"
  local marker="$2"
  shift 2
  local index
  index="$(stage_index "$name")"
  if (( index < FROM_INDEX || index > TO_INDEX )); then
    return 0
  fi
  CURRENT_STAGE="$name"
  echo
  echo "=== $name ==="
  print_command "$@"
  if (( RESUME == 1 )) && [[ -n "$marker" && -f "$marker" ]]; then
    echo "SKIP existing completion artifact: $marker"
    return 0
  fi
  if (( DRY_RUN == 1 )); then
    return 0
  fi
  "$@"
}

preflight
if (( CHECK_ONLY == 1 )); then
  echo "Preflight passed; no data, checkpoints, or results were changed."
  exit 0
fi

split_cmd=(
  "$PYTHON_BIN" -u scripts/create_local_holdout.py
  --dataset_dir dataset_train
  --test_ratio "$TEST_RATIO"
  --min_test_per_category "$MIN_TEST_PER_CATEGORY"
  --seed "$SEED"
)
(( OVERWRITE_SPLIT == 1 )) && split_cmd+=(--overwrite)
run_stage prepare-split dataset_train/local_split_manifest.json "${split_cmd[@]}"

cache_cmd=(
  "$PYTHON_BIN" -u scripts/precompute_clean_points.py
  --mode cache
  --input_dir dataset_train
  --output_dir dataset_train_pcd_disk
  --splits local_train local_test
  --train_num_points "$TRAIN_NUM_POINTS"
  --test_num_points "$TEST_NUM_POINTS"
  --workers "$WORKERS"
  --seed "$SEED"
)
(( OVERWRITE_CACHE == 1 )) && cache_cmd+=(--overwrite)
run_stage prepare-cache dataset_train_pcd_disk/precompute_manifest.json "${cache_cmd[@]}"

run_stage train-vm experiments/vm/checkpoint_99.pkl \
  "$PYTHON_BIN" -u run.py --task configs/task/train_vm_cached.yaml --seed "$SEED"

selection_cmd() {
  local stage_name="$1"
  local checkpoint_dir="$2"
  local task="$3"
  local output_dir="$4"
  local command=(
    "$PYTHON_BIN" -u select_best_checkpoint.py
    --metric composite
    --ckpt_dir "$checkpoint_dir"
    --task_template "$task"
    --mesh_dir dataset_train/local_train
    --output_dir "$output_dir"
    --prefilter_top_k "$PREFILTER_TOP_K"
    --prefilter_last_n "$PREFILTER_LAST_N"
    --prefilter_cd_points "$PREFILTER_CD_POINTS"
    --prefilter_cd_limit "$PREFILTER_CD_LIMIT"
    --validation_workers 0
    --use_cuda "$USE_CUDA"
    --seed "$SEED"
    --copy_best
  )
  (( RESUME_SELECTION == 1 )) && command+=(--resume)
  run_stage "$stage_name" "$output_dir/best_checkpoint.pkl" "${command[@]}"
}

selection_cmd select-vm experiments/vm configs/task/train_vm_cached.yaml checkpoint_selection_cached

run_stage train-cvm experiments/cvm/checkpoint_99.pkl \
  "$PYTHON_BIN" -u run.py --task configs/task/train_cvm_cached.yaml --seed "$SEED"

selection_cmd select-cvm experiments/cvm configs/task/train_cvm_cached.yaml checkpoint_selection_cvm

run_stage train-straightpcf experiments/straightpcf/checkpoint_99.pkl \
  "$PYTHON_BIN" -u run.py --task configs/task/train_straightpcf_cached.yaml --seed "$SEED"

selection_cmd select-straightpcf experiments/straightpcf configs/task/train_straightpcf_cached.yaml checkpoint_selection_straightpcf

evaluate_cmd=(
  "$PYTHON_BIN" -u scripts/evaluate_local_test_models.py
  --model VM vm checkpoint_selection_cached/best_checkpoint.pkl
  --model CVM cvm checkpoint_selection_cvm/best_checkpoint.pkl
  --model StraightPCF straightpcf checkpoint_selection_straightpcf/best_checkpoint.pkl
  --data-root dataset_train_pcd_disk/local_test
  --mesh-root dataset_train/local_test
  --datalist dataset_train/local_test/datalist.txt
  --result-file "$RESULT_FILE"
  --fusion-mode "$FUSION_MODE"
  --use-cuda "$USE_CUDA"
  --seed "$SEED"
)
[[ -n "$EVAL_LIMIT" ]] && evaluate_cmd+=(--limit "$EVAL_LIMIT")
run_stage evaluate "" "${evaluate_cmd[@]}"

echo
echo "Pipeline completed through stage: $TO_STAGE"
echo "Local-test results: $RESULT_FILE"
