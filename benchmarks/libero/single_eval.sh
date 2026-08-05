#!/usr/bin/env bash
# Run one LIBERO suite/task selection against an already-running sana-wam
# policy server. The LIBERO simulator stays in its own Python environment.
#
# Usage:
#   bash single_eval.sh <suite> <task_id|all> <num_trials> <gpu_id> [http_port] [host]
#
# Required environment variables:
#   LIBERO_PATH         LIBERO checkout root (contains libero/)
#   LIBERO_PYTHON       Python executable from the dedicated LIBERO environment
#   LIBERO_CONFIG_PATH  Existing directory containing LIBERO config.yaml
#
# Optional environment variables:
#   LIBERO_OUTPUT_DIR     Result directory (default: benchmarks/libero/results)
#   LIBERO_POLICY_CONFIG  Alternate client config
#   LIBERO_EGL_DEVICE_ID  EGL device visible inside the simulator process
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "Usage: bash single_eval.sh <suite> <task_id|all> <num_trials> <gpu_id> [http_port] [host]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LIBERO_PATH="${LIBERO_PATH:?LIBERO_PATH must be set to the LIBERO checkout root}"
LIBERO_PYTHON="${LIBERO_PYTHON:?LIBERO_PYTHON must be set to the dedicated LIBERO Python executable}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:?LIBERO_CONFIG_PATH must be set to a directory containing config.yaml}"
EXPECTED_LIBERO_GIT_SHA="8f1084e3132a39270c3a13ebe37270a43ece2a01"
TASK_ORDER_INDEX=0
ENVIRONMENT_SEED=0
CAMERA_HEIGHT=256
CAMERA_WIDTH=256

[[ -d "${LIBERO_PATH}" && -d "${LIBERO_PATH}/libero" ]] || {
    echo "[ERROR] LIBERO_PATH must contain the libero/ package: ${LIBERO_PATH}" >&2
    exit 1
}
[[ -x "${LIBERO_PYTHON}" ]] || {
    echo "[ERROR] LIBERO_PYTHON is not an executable file: ${LIBERO_PYTHON}" >&2
    exit 1
}
[[ -d "${LIBERO_CONFIG_PATH}" && -f "${LIBERO_CONFIG_PATH}/config.yaml" ]] || {
    echo "[ERROR] LIBERO_CONFIG_PATH must be an existing directory containing config.yaml: ${LIBERO_CONFIG_PATH}" >&2
    exit 1
}

actual_libero_git_sha="$(git -C "${LIBERO_PATH}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${actual_libero_git_sha}" != "${EXPECTED_LIBERO_GIT_SHA}" ]]; then
    echo "[ERROR] LIBERO checkout HEAD mismatch: expected ${EXPECTED_LIBERO_GIT_SHA}, got ${actual_libero_git_sha:-<not-a-git-checkout>}." >&2
    exit 1
fi
libero_worktree_status="$(git -C "${LIBERO_PATH}" status --porcelain=v1 --untracked-files=all)"
if [[ -n "${libero_worktree_status}" ]]; then
    echo "[ERROR] LIBERO checkout is dirty despite the pinned HEAD:" >&2
    echo "${libero_worktree_status}" >&2
    exit 1
fi

eval_script="${SCRIPT_DIR}/eval_policy.py"
policy_config="${LIBERO_POLICY_CONFIG:-${SCRIPT_DIR}/policy_config.yml}"
output_dir="${LIBERO_OUTPUT_DIR:-${SCRIPT_DIR}/results}"

[[ -f "${eval_script}" ]] || {
    echo "[ERROR] LIBERO evaluation runner not found: ${eval_script}" >&2
    exit 1
}
[[ -f "${policy_config}" ]] || {
    echo "[ERROR] LIBERO policy config not found: ${policy_config}" >&2
    exit 1
}
[[ -n "${output_dir}" ]] || {
    echo "[ERROR] LIBERO_OUTPUT_DIR must not be empty" >&2
    exit 1
}

suite="$1"
task_id="$2"
num_trials="$3"
gpu_id="$4"
http_port="${5:-${LIBERO_HTTP_PORT:-8848}}"
host="${6:-${LIBERO_POLICY_HOST:-127.0.0.1}}"

case "${suite}" in
    libero_spatial|libero_object|libero_goal|libero_10)
        suite_task_count=10
        ;;
    libero_90)
        echo "[ERROR] libero_90 is dataset/pretraining-only in this adapter and is not an allowed closed-loop suite." >&2
        exit 1
        ;;
    *)
        echo "[ERROR] Unsupported suite '${suite}'. Expected libero_spatial, libero_object, libero_goal, or libero_10." >&2
        exit 1
        ;;
esac

if [[ "${task_id}" != "all" ]]; then
    if ! [[ "${task_id}" =~ ^[0-9]+$ ]] || (( 10#${task_id} >= suite_task_count )); then
        echo "[ERROR] task_id must be 'all' or an integer in [0, $((suite_task_count - 1))] for ${suite}; got '${task_id}'." >&2
        exit 1
    fi
fi
if ! [[ "${num_trials}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] num_trials must be a positive integer; got '${num_trials}'." >&2
    exit 1
fi
if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] gpu_id must be a non-negative physical GPU index; got '${gpu_id}'." >&2
    exit 1
fi
if ! [[ "${http_port}" =~ ^[0-9]+$ ]] || (( http_port < 1 || http_port > 65535 )); then
    echo "[ERROR] http_port must be in [1, 65535]; got '${http_port}'." >&2
    exit 1
fi
if ! [[ "${host}" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
    echo "[ERROR] host contains unsupported characters: '${host}'." >&2
    exit 1
fi

echo "suite         : ${suite}"
echo "task id       : ${task_id}"
echo "trials/task   : ${num_trials}"
echo "simulator GPU : ${gpu_id}"
echo "policy server : http://${host}:${http_port}"
echo "output dir    : ${output_dir}"
echo "LIBERO path   : ${LIBERO_PATH}"
echo "LIBERO SHA    : ${actual_libero_git_sha}"
echo "LIBERO config : ${LIBERO_CONFIG_PATH}/config.yaml"
echo "task order    : ${TASK_ORDER_INDEX}"
echo "env seed      : ${ENVIRONMENT_SEED}"
echo "camera        : ${CAMERA_HEIGHT}x${CAMERA_WIDTH}"

# CUDA_VISIBLE_DEVICES isolates the simulator to the selected physical GPU;
# inside that process its EGL device is normally logical device 0. The policy
# server is external and is neither started nor assigned a GPU by this script.
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${LIBERO_EGL_DEVICE_ID:-0}"
export LIBERO_CONFIG_PATH
export PYTHONPATH="${LIBERO_PATH}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"

PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
"${LIBERO_PYTHON}" "${eval_script}" \
    --config "${policy_config}" \
    --task-suite-name "${suite}" \
    --task-id "${task_id}" \
    --num-trials-per-task "${num_trials}" \
    --task-order-index "${TASK_ORDER_INDEX}" \
    --seed "${ENVIRONMENT_SEED}" \
    --camera-height "${CAMERA_HEIGHT}" \
    --camera-width "${CAMERA_WIDTH}" \
    --host "${host}" \
    --http-port "${http_port}" \
    --output-dir "${output_dir}"
