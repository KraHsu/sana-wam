#!/usr/bin/env bash
# Run the native sana-wam AR low-noise checkpoint through RoboTwin episodes.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-${ROOT}/logs/native_ar_lownoise_20260714}"
HTTP_PORT="${HTTP_PORT:-18620}"
SIM_GPU="${SIM_GPU:-6}"
TEST_NUM="${ROBOTWIN_TEST_NUM:-100}"
LABEL="${LABEL:-sana_native_ar_lownoise_seedfixed_n${TEST_NUM}}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/home/zch/RoboTwin}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/home/zch/miniconda3/envs/RoboTwin/bin/python}"
ROBOTWIN_POLICY_CONFIG_PATH="${ROBOTWIN_POLICY_CONFIG_PATH:-${ROOT}/benchmarks/robotwin/policy_config.yml}"

[[ -f "${ROBOTWIN_POLICY_CONFIG_PATH}" ]] || {
    echo "Policy config not found: ${ROBOTWIN_POLICY_CONFIG_PATH}" >&2
    exit 2
}

mkdir -p \
    "${RUN_DIR}/robotwin_runtime" \
    "${RUN_DIR}/tmp" \
    "${RUN_DIR}/matplotlib"

# Canonical evaluation uses RoboTwin's own task limits. Refuse to start if a
# repository-side override is accidentally re-enabled.
if rg -q '^[[:space:]]*[A-Za-z0-9_]+:[[:space:]]*[0-9]+' \
    "${ROOT}/benchmarks/robotwin/step_limits.yml"; then
    echo "Active RoboTwin step-limit override found; refusing canonical evaluation." >&2
    exit 2
fi

if ! curl -fsS "http://127.0.0.1:${HTTP_PORT}/health" > "${RUN_DIR}/health.before.json"; then
    echo "Policy server is not healthy on HTTP port ${HTTP_PORT}." >&2
    exit 3
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/started.utc"

set +e
env -u ROBOTWIN_STEP_LIMITS_PATH \
    ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
    ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON}" \
    ROBOTWIN_TEST_NUM="${TEST_NUM}" \
    ROBOTWIN_RUNTIME_ROOT="${RUN_DIR}/robotwin_runtime" \
    ROBOTWIN_ENABLE_PLANNER_FALLBACK=0 \
    POLICY_CONFIG_PATH="${ROBOTWIN_POLICY_CONFIG_PATH}" \
    TMPDIR="${RUN_DIR}/tmp" \
    MPLCONFIGDIR="${RUN_DIR}/matplotlib" \
    bash "${ROOT}/benchmarks/robotwin/single_eval.sh" \
        adjust_bottle demo_clean "${LABEL}" "${SIM_GPU}" "${HTTP_PORT}" 127.0.0.1 \
        > "${RUN_DIR}/eval.log" 2>&1
status=$?
set -e

printf '%s\n' "${status}" > "${RUN_DIR}/eval.exit"
date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/finished.utc"
exit "${status}"
