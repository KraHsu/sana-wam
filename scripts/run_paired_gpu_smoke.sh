#!/usr/bin/env bash
# Run predicted and one cache-feedback treatment on separate GPUs and verify pairing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREDICTED_DIR="${PREDICTED_DIR:-/tmp/sana_wam_paired_predicted_smoke_v3}"
TREATMENT_MODE="${TREATMENT_MODE:-measured}"
case "${TREATMENT_MODE}" in
    predicted | reencode_predicted | measured) ;;
    *)
        echo "Unsupported TREATMENT_MODE: ${TREATMENT_MODE}" >&2
        exit 2
        ;;
esac
TREATMENT_DIR="${TREATMENT_DIR:-${MEASURED_DIR:-/tmp/sana_wam_paired_${TREATMENT_MODE}_smoke_v3}}"
PREDICTED_GPU="${PREDICTED_GPU:-0}"
TREATMENT_GPU="${TREATMENT_GPU:-${MEASURED_GPU:-1}}"
CHECKPOINT_SHA="${CHECKPOINT_SHA:-aea6b58c9107aeeeffba561f39ffedfc3f8fd8420a4abfcab9038c8cdd4a129d}"

for output_dir in "${PREDICTED_DIR}" "${TREATMENT_DIR}"; do
    if [[ -e "${output_dir}" ]]; then
        echo "Smoke output already exists: ${output_dir}" >&2
        exit 2
    fi
    mkdir -p "${output_dir}"
done

env \
    SANA_WAM_TELEMETRY_DIR="${PREDICTED_DIR}/telemetry" \
    SANA_WAM_TELEMETRY_RUN_ID="paired_gpu_smoke_predicted" \
    SANA_WAM_CHECKPOINT_SHA256="${CHECKPOINT_SHA}" \
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/smoke_ar_lownoise_gpu.py" \
        --deploy-config "${ROOT}/configs/deploy_ar_lownoise_paired_predicted.yaml" \
        --device "cuda:${PREDICTED_GPU}" \
        > "${PREDICTED_DIR}/smoke.log" 2>&1 &
predicted_pid=$!

env \
    SANA_WAM_TELEMETRY_DIR="${TREATMENT_DIR}/telemetry" \
    SANA_WAM_TELEMETRY_RUN_ID="paired_gpu_smoke_${TREATMENT_MODE}" \
    SANA_WAM_CHECKPOINT_SHA256="${CHECKPOINT_SHA}" \
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/smoke_ar_lownoise_gpu.py" \
        --deploy-config "${ROOT}/configs/deploy_ar_lownoise_paired_${TREATMENT_MODE}.yaml" \
        --device "cuda:${TREATMENT_GPU}" \
        > "${TREATMENT_DIR}/smoke.log" 2>&1 &
treatment_pid=$!

set +e
wait "${predicted_pid}"
predicted_status=$?
wait "${treatment_pid}"
treatment_status=$?
set -e

if (( predicted_status != 0 || treatment_status != 0 )); then
    echo "GPU smoke failed: predicted=${predicted_status} ${TREATMENT_MODE}=${treatment_status}" >&2
    tail -n 40 "${PREDICTED_DIR}/smoke.log" >&2
    tail -n 40 "${TREATMENT_DIR}/smoke.log" >&2
    exit 3
fi

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/verify_paired_noise_telemetry.py" \
    "${PREDICTED_DIR}/telemetry" "${TREATMENT_DIR}/telemetry" \
    --expected-episodes 1 \
    --control-mode predicted \
    --treatment-mode "${TREATMENT_MODE}"

tail -n 8 "${PREDICTED_DIR}/smoke.log"
tail -n 8 "${TREATMENT_DIR}/smoke.log"
