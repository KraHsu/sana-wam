#!/usr/bin/env bash
# Run a single RoboTwin task evaluation against an already-running sana-wam server.
#
# Usage:
#   bash single_eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [http_port] [host]
#
# Args:
#   task_name    — RoboTwin task (e.g. adjust_bottle)
#   task_config  — demo_clean | demo_randomized
#   ckpt_setting — label used in result filenames (e.g. sana_wam)
#   gpu_id       — CUDA device for the RoboTwin simulator process
#   http_port    — sana-wam HTTP port  (default: 8848, env: ROBOTWIN_HTTP_PORT)
#   host         — sana-wam server host (default: 127.0.0.1, env: ROBOTWIN_POLICY_HOST)
#
# Required env vars:
#   ROBOTWIN_PATH    — path to the RoboTwin repository
#   ROBOTWIN_PYTHON  — Python interpreter for the RoboTwin env
# Optional env vars:
#   ROBOTWIN_TEST_NUM — cap RoboTwin eval episodes for smoke runs (default: upstream 100)
#   ROBOTWIN_ENV_SEED_INDEX — RoboTwin start-seed index (default: 0 -> 100000;
#                             use 1 -> 200000 for training-data collection)
#   ROBOTWIN_ENV_SEED_OFFSET — offset within the selected 100xxx block
#   ROBOTWIN_STEP_LIMITS_PATH — alternate task_name->step_lim YAML
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: bash single_eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [http_port] [host]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROBOTWIN_PATH="${ROBOTWIN_PATH:?ROBOTWIN_PATH must be set to the RoboTwin repository root}"
[[ -d "${ROBOTWIN_PATH}" ]] || { echo "[ERROR] ROBOTWIN_PATH not found: ${ROBOTWIN_PATH}" >&2; exit 1; }

robotwin_eval_script="${SCRIPT_DIR}/eval_policy_wrapper.py"
[[ -f "${robotwin_eval_script}" ]] || { echo "[ERROR] eval wrapper not found: ${robotwin_eval_script}" >&2; exit 1; }

task_name="$1"
task_config="$2"
ckpt_setting="${3:-sana_wam}"
gpu_id="${4:-0}"
http_port="${5:-${ROBOTWIN_HTTP_PORT:-8848}}"
host="${6:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
seed="${ROBOTWIN_ENV_SEED_INDEX:-0}"
seed_offset="${ROBOTWIN_ENV_SEED_OFFSET:-0}"

robotwin_python="${ROBOTWIN_PYTHON:-python}"
policy_config_template="${POLICY_CONFIG_PATH:-${SCRIPT_DIR}/policy_config.yml}"

[[ -f "${policy_config_template}" ]] || {
    echo "[ERROR] policy_config.yml not found: ${policy_config_template}" >&2; exit 1; }

PYTHONDONTWRITEBYTECODE=1 \
"${SANA_WAM_GUARD_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}" \
    "${PROJECT_ROOT}/scripts/check_cach_stage0_reserved_config.py" \
    "${policy_config_template}" \
    --entrypoint "benchmarks/robotwin/single_eval.sh policy config"

if ! [[ "${task_name}" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "[ERROR] Invalid task_name '${task_name}'. Expected [A-Za-z0-9_]+." >&2
    exit 1
fi
if [[ "${task_config}" != "demo_clean" && "${task_config}" != "demo_randomized" ]]; then
    echo "[ERROR] Invalid task_config '${task_config}'." >&2
    exit 1
fi
if ! [[ "${http_port}" =~ ^[0-9]+$ ]] || (( http_port < 1 || http_port > 65535 )); then
    echo "[ERROR] Invalid http_port '${http_port}'. Expected 1..65535." >&2
    exit 1
fi
if ! [[ "${host}" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
    echo "[ERROR] Invalid host '${host}'. Expected hostname/IP characters only." >&2
    exit 1
fi
if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid ROBOTWIN_ENV_SEED_INDEX '${seed}'. Expected a non-negative integer." >&2
    exit 1
fi
if ! [[ "${seed_offset}" =~ ^[0-9]+$ ]] || (( 10#${seed_offset} >= 100000 )); then
    echo "[ERROR] Invalid ROBOTWIN_ENV_SEED_OFFSET '${seed_offset}'. Expected an integer in [0, 100000)." >&2
    exit 1
fi

maybe_configure_sapien_egl() {
    [[ -n "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" || -n "${__EGL_VENDOR_LIBRARY_DIRS:-}" ]] && return 0

    local egl_json
    egl_json="$(dirname "$(dirname "${robotwin_python}")")/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
    [[ -n "${egl_json}" && -f "${egl_json}" ]] || return 0

    # SAPIEN's import-time EGL probe crashes on some cluster images because it
    # blindly lists /usr/share/glvnd/egl_vendor.d when that directory is absent.
    export __EGL_VENDOR_LIBRARY_FILENAMES="${egl_json}"
    echo "[INFO] SAPIEN EGL ICD: ${__EGL_VENDOR_LIBRARY_FILENAMES}"
}

# Inject runtime host and http_port into a temp config
runtime_config="$(mktemp "${TMPDIR:-/tmp}/sana_wam_policy_config.XXXXXX.yml")"
trap 'rm -f "${runtime_config}"' EXIT

sed \
    -e "s/^host:.*/host: \"${host}\"/" \
    -e "s/^http_port:.*/http_port: ${http_port}/" \
    "${policy_config_template}" > "${runtime_config}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
# PYTHONPATH: RoboTwin modules + this directory (for sana_wam2robotwin_interface.py)
export PYTHONPATH="${ROBOTWIN_PATH}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
maybe_configure_sapien_egl

cd "${ROBOTWIN_PATH}"

echo "task_name    : ${task_name}"
echo "task_config  : ${task_config}"
echo "ckpt_setting : ${ckpt_setting}"
echo "server       : http://${host}:${http_port}"
echo "gpu          : ${gpu_id}"
echo "seed         : ${seed}"
echo "seed offset  : ${seed_offset}"
echo "step limits  : ${ROBOTWIN_STEP_LIMITS_PATH:-${SCRIPT_DIR}/step_limits.yml}"

PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning \
"${robotwin_python}" "${robotwin_eval_script}" \
    --config    "${runtime_config}" \
    --overrides \
    --task_name        "${task_name}" \
    --task_config      "${task_config}" \
    --ckpt_setting     "${ckpt_setting}" \
    --seed             "${seed}" \
    --policy_name      "sana_wam2robotwin_interface"
