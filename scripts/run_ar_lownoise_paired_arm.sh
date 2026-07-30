#!/usr/bin/env bash
# Launch one common-random-number evaluation or replay arm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${1:-}"
case "${ARM}" in
    predicted | reencode_predicted | measured) ;;
    *)
        echo "Usage: bash scripts/run_ar_lownoise_paired_arm.sh predicted|reencode_predicted|measured" >&2
        exit 2
        ;;
esac
if [[ "${ARM}" != "predicted" && -z "${PAIR_WITH:-}" ]]; then
    echo "${ARM} arm requires PAIR_WITH=<reference run dir> for mandatory pairing validation." >&2
    exit 2
fi
PAIR_REFERENCE_MODE="${PAIR_REFERENCE_MODE:-predicted}"
case "${PAIR_REFERENCE_MODE}" in
    predicted | reencode_predicted | measured) ;;
    *)
        echo "Unsupported PAIR_REFERENCE_MODE: ${PAIR_REFERENCE_MODE}" >&2
        exit 2
        ;;
esac
PAIR_COMPARISON_KIND="${PAIR_COMPARISON_KIND:-noise}"
case "${PAIR_COMPARISON_KIND}" in
    noise) ;;
    model)
        if [[ -z "${PAIR_WITH:-}" ]]; then
            echo "PAIR_COMPARISON_KIND=model requires PAIR_WITH=<control run dir>." >&2
            exit 2
        fi
        if [[ "${PAIR_REFERENCE_MODE}" != "predicted" || "${ARM}" != "predicted" ]]; then
            echo "PAIR_COMPARISON_KIND=model requires predicted control and treatment arms." >&2
            exit 2
        fi
        ;;
    rerank)
        if [[ -z "${PAIR_WITH:-}" ]]; then
            echo "PAIR_COMPARISON_KIND=rerank requires PAIR_WITH=<control run dir>." >&2
            exit 2
        fi
        if [[ "${PAIR_REFERENCE_MODE}" != "predicted" || "${ARM}" != "predicted" ]]; then
            echo "PAIR_COMPARISON_KIND=rerank requires predicted control and treatment arms." >&2
            exit 2
        fi
        ;;
    *)
        echo "Unsupported PAIR_COMPARISON_KIND: ${PAIR_COMPARISON_KIND}" >&2
        exit 2
        ;;
esac

TEST_NUM="${ROBOTWIN_TEST_NUM:-30}"
RUN_DIR="${RUN_DIR:-${ROOT}/logs/paired_${ARM}_n${TEST_NUM}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$(readlink -m "${RUN_DIR}")"
INPUT_PROMPT_MANIFEST="${ROBOTWIN_PROMPT_MANIFEST:-}"
HTTP_PORT="${HTTP_PORT:-18630}"
WS_PORT="${WS_PORT:-$((HTTP_PORT + 1))}"
SERVER_GPU="${SERVER_GPU:-0}"
SIM_GPU="${SIM_GPU:-1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/zch/wuji-openwam-dev/sandbox/sana_ar_graft_AC_lownoise_only/2026-07-04_13-03-25}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_step_12000.safetensors}"
EXPECTED_CHECKPOINT_SHA="${EXPECTED_CHECKPOINT_SHA:-aea6b58c9107aeeeffba561f39ffedfc3f8fd8420a4abfcab9038c8cdd4a129d}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-${ROOT}/configs/deploy_ar_lownoise_paired_${ARM}.yaml}"
EXPECTED_PROMPT_MANIFEST_SHA256="${EXPECTED_PROMPT_MANIFEST_SHA256:-}"
EXPECTED_RANKER_SHA256="${EXPECTED_RANKER_SHA256:-c5af1d0d6c145e06c127b4f0d9292a5a05d523ed1a0317d24b482a3f6d5cc964}"
RERANK_CANDIDATE_COUNT="${RERANK_CANDIDATE_COUNT:-5}"

RUN_LOCK="${RUN_DIR}.run-lock"
if ! mkdir "${RUN_LOCK}" 2>/dev/null; then
    echo "Run lock already exists; refusing a concurrent duplicate: ${RUN_LOCK}" >&2
    exit 3
fi
release_lock() {
    rmdir "${RUN_LOCK}" 2>/dev/null || true
}
trap release_lock EXIT

if [[ -n "${PAIR_WITH:-}" ]]; then
    [[ -d "${PAIR_WITH}" ]] || {
        echo "Pair reference run not found: ${PAIR_WITH}" >&2
        exit 2
    }
    PAIR_WITH="$(readlink -f "${PAIR_WITH}")"
fi
[[ -f "${DEPLOY_CONFIG}" ]] || {
    echo "Deploy config not found: ${DEPLOY_CONFIG}" >&2
    exit 2
}

if [[ -d "${RUN_DIR}" ]]; then
    shopt -s nullglob dotglob
    existing=("${RUN_DIR}"/*)
    shopt -u nullglob dotglob
    if (( ${#existing[@]} > 0 )); then
        echo "RUN_DIR is not empty; refusing to append/overwrite telemetry: ${RUN_DIR}" >&2
        exit 3
    fi
fi

if rg -q '^[[:space:]]*[A-Za-z0-9_]+:[[:space:]]*[0-9]+' \
    "${ROOT}/benchmarks/robotwin/step_limits.yml"; then
    echo "Active RoboTwin step-limit override found; refusing paired evaluation." >&2
    exit 4
fi

if curl -fsS "http://127.0.0.1:${HTTP_PORT}/health" > /dev/null 2>&1; then
    echo "HTTP port ${HTTP_PORT} already has a healthy server; refusing stale-server reuse." >&2
    exit 5
fi

checkpoint_path="${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"
[[ -f "${checkpoint_path}" ]] || {
    echo "Checkpoint not found: ${checkpoint_path}" >&2
    exit 6
}
checkpoint_sha="$(sha256sum "${checkpoint_path}" | awk '{print $1}')"
if [[ "${checkpoint_sha}" != "${EXPECTED_CHECKPOINT_SHA}" ]]; then
    echo "Checkpoint SHA-256 mismatch: ${checkpoint_sha}" >&2
    exit 7
fi
mkdir -p "${RUN_DIR}/telemetry/server" "${RUN_DIR}/telemetry/client"
printf '%s  %s\n' "${checkpoint_sha}" "${checkpoint_path}" > "${RUN_DIR}/checkpoint.sha256"

prompt_manifest_path=""
if [[ -n "${PAIR_WITH:-}" || -n "${ROBOTWIN_PROMPT_MANIFEST:-}" ]]; then
    prompt_manifest_path="${RUN_DIR}/prompt_manifest.json"
    if [[ -n "${ROBOTWIN_PROMPT_MANIFEST:-}" ]]; then
        [[ -f "${ROBOTWIN_PROMPT_MANIFEST}" ]] || {
            echo "Prompt manifest not found: ${ROBOTWIN_PROMPT_MANIFEST}" >&2
            exit 8
        }
        manifest_source="$(readlink -f "${ROBOTWIN_PROMPT_MANIFEST}")"
        manifest_destination="$(readlink -m "${prompt_manifest_path}")"
        if [[ "${manifest_source}" != "${manifest_destination}" ]]; then
            cp "${manifest_source}" "${prompt_manifest_path}"
        fi
    elif [[ -n "${PAIR_WITH:-}" ]]; then
        if [[ "${PAIR_REFERENCE_MODE}" != "predicted" ]]; then
            echo "ROBOTWIN_PROMPT_MANIFEST is required when the pair reference mode is ${PAIR_REFERENCE_MODE}." >&2
            exit 8
        fi
        "${ROOT}/.venv/bin/python" \
            "${ROOT}/scripts/build_robotwin_prompt_manifest.py" \
            "${PAIR_WITH}" "${prompt_manifest_path}" \
            --minimum-episodes "${TEST_NUM}" \
            --instruction-type unseen
    else
        echo "Internal error: prompt-manifest setup has no source." >&2
        exit 8
    fi
    prompt_manifest_sha256="$(sha256sum "${prompt_manifest_path}" | awk '{print $1}')"
    if [[ -n "${EXPECTED_PROMPT_MANIFEST_SHA256}" && \
        "${prompt_manifest_sha256}" != "${EXPECTED_PROMPT_MANIFEST_SHA256}" ]]; then
        echo "Prompt manifest SHA-256 mismatch: ${prompt_manifest_sha256}" >&2
        exit 8
    fi
    printf '%s  %s\n' \
        "${prompt_manifest_sha256}" "${prompt_manifest_path}" \
        > "${RUN_DIR}/prompt_manifest.sha256"
fi

SANA_WAM_TELEMETRY_DIR="${RUN_DIR}/telemetry/server" \
SANA_WAM_TELEMETRY_RUN_ID="paired_${ARM}_n${TEST_NUM}" \
SANA_WAM_CHECKPOINT_SHA256="${checkpoint_sha}" \
PYTHONUNBUFFERED=1 \
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/deploy.py" \
    --ckpt-dir "${CHECKPOINT_DIR}" \
    --ckpt-name "${CHECKPOINT_NAME}" \
    --deploy-config "${DEPLOY_CONFIG}" \
    --device "cuda:${SERVER_GPU}" \
    --host 127.0.0.1 \
    --http-port "${HTTP_PORT}" \
    --ws-port "${WS_PORT}" \
    > "${RUN_DIR}/server.log" 2>&1 &
server_pid=$!

cleanup() {
    if kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}"
        wait "${server_pid}" || true
    fi
    release_lock
}
trap cleanup EXIT

for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:${HTTP_PORT}/health" > /dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "Policy server exited before becoming healthy; see ${RUN_DIR}/server.log" >&2
        exit 8
    fi
    sleep 1
done
curl -fsS "http://127.0.0.1:${HTTP_PORT}/info" > "${RUN_DIR}/info.before.json"

"${ROOT}/.venv/bin/python" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); r=d["inference_runtime"]; i=r["deployment_identity"]; assert r["episode_noise_mode"]=="paired", r; assert r["cache_feedback_mode"]==sys.argv[2], r; assert i["checkpoint_sha256"]==sys.argv[3], i; assert i["history_len"]==113 and i["video_steps"]==10 and i["action_steps"]==10 and i["action_tokens_per_chunk"]==28, i' \
    "${RUN_DIR}/info.before.json" "${ARM}" "${checkpoint_sha}"

RUN_DIR="${RUN_DIR}" \
HTTP_PORT="${HTTP_PORT}" \
SIM_GPU="${SIM_GPU}" \
ROBOTWIN_TEST_NUM="${TEST_NUM}" \
LABEL="${LABEL:-sana_ar_paired_${ARM}_n${TEST_NUM}}" \
ROBOTWIN_TELEMETRY_DIR="${RUN_DIR}/telemetry/client" \
SANA_WAM_TELEMETRY_RUN_ID="paired_${ARM}_n${TEST_NUM}" \
ROBOTWIN_PROMPT_MANIFEST="${prompt_manifest_path}" \
bash "${ROOT}/scripts/eval_ar_lownoise_seedfixed.sh"

curl -fsS "http://127.0.0.1:${HTTP_PORT}/info" > "${RUN_DIR}/info.after.json"
cleanup
trap - EXIT

if [[ -z "${PAIR_WITH:-}" && -z "${INPUT_PROMPT_MANIFEST}" ]]; then
    prompt_manifest_path="${RUN_DIR}/prompt_manifest.json"
    "${ROOT}/.venv/bin/python" \
        "${ROOT}/scripts/build_robotwin_prompt_manifest.py" \
        "${RUN_DIR}" "${prompt_manifest_path}" \
        --expected-episodes "${TEST_NUM}" \
        --instruction-type unseen
    prompt_manifest_sha256="$(sha256sum "${prompt_manifest_path}" | awk '{print $1}')"
    printf '%s  %s\n' \
        "${prompt_manifest_sha256}" "${prompt_manifest_path}" \
        > "${RUN_DIR}/prompt_manifest.sha256"
fi

if [[ -n "${PAIR_WITH:-}" && "${SKIP_PAIR_VERIFY:-0}" != "1" ]]; then
    pair_path="${PAIR_WITH}"
    if [[ "${PAIR_COMPARISON_KIND}" == "model" ]]; then
        verify_args=(
            "${pair_path}" "${RUN_DIR}"
            --expected-episodes "${TEST_NUM}"
            --control-client "${pair_path}"
            --treatment-client "${RUN_DIR}"
        )
        if [[ -n "${EXPECTED_PROMPT_MANIFEST_SHA256}" ]]; then
            verify_args+=(
                --expected-prompt-manifest-sha256
                "${EXPECTED_PROMPT_MANIFEST_SHA256}"
            )
        fi
        if [[ "${PAIR_ALLOW_CONTROL_PREFIX:-0}" == "1" ]]; then
            verify_args+=(--allow-control-prefix)
        fi
        "${ROOT}/.venv/bin/python" \
            "${ROOT}/scripts/verify_paired_model_telemetry.py" \
            "${verify_args[@]}"
    elif [[ "${PAIR_COMPARISON_KIND}" == "rerank" ]]; then
        verify_args=(
            "${pair_path}" "${RUN_DIR}"
            --expected-episodes "${TEST_NUM}"
            --expected-ranker-sha256 "${EXPECTED_RANKER_SHA256}"
            --candidate-count "${RERANK_CANDIDATE_COUNT}"
        )
        if [[ -n "${EXPECTED_PROMPT_MANIFEST_SHA256}" ]]; then
            verify_args+=(
                --expected-prompt-manifest-sha256
                "${EXPECTED_PROMPT_MANIFEST_SHA256}"
            )
        fi
        if [[ "${PAIR_ALLOW_CONTROL_PREFIX:-0}" == "1" ]]; then
            verify_args+=(--allow-control-prefix)
        fi
        "${ROOT}/.venv/bin/python" \
            "${ROOT}/scripts/verify_paired_rerank_telemetry.py" \
            "${verify_args[@]}"
    else
        verify_args=(
            "${pair_path}" "${RUN_DIR}"
            --expected-episodes "${TEST_NUM}"
            --control-client "${pair_path}"
            --treatment-client "${RUN_DIR}"
            --control-mode "${PAIR_REFERENCE_MODE}"
            --treatment-mode "${ARM}"
        )
        if [[ -n "${EXPECTED_PROMPT_MANIFEST_SHA256}" ]]; then
            verify_args+=(
                --expected-prompt-manifest-sha256
                "${EXPECTED_PROMPT_MANIFEST_SHA256}"
            )
        fi
        if [[ "${PAIR_ALLOW_CONTROL_PREFIX:-0}" == "1" ]]; then
            verify_args+=(--allow-control-prefix)
        fi
        "${ROOT}/.venv/bin/python" \
            "${ROOT}/scripts/verify_paired_noise_telemetry.py" \
            "${verify_args[@]}"
    fi
fi
