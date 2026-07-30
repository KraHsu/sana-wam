#!/usr/bin/env bash
# Run one real RoboTwin episode per arm, including client telemetry and strict verification.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREDICTED_RUN="${PREDICTED_RUN:-/tmp/sana_wam_robotwin_predicted_smoke_v1}"
MEASURED_RUN="${MEASURED_RUN:-/tmp/sana_wam_robotwin_measured_smoke_v1}"
HTTP_PORT="${HTTP_PORT:-18640}"
SERVER_GPU="${SERVER_GPU:-0}"
SIM_GPU="${SIM_GPU:-1}"

if [[ -e "${PREDICTED_RUN}" || -e "${MEASURED_RUN}" ]]; then
    echo "RoboTwin smoke output already exists; choose fresh PREDICTED_RUN/MEASURED_RUN." >&2
    exit 2
fi

ROBOTWIN_TEST_NUM=1 \
RUN_DIR="${PREDICTED_RUN}" \
HTTP_PORT="${HTTP_PORT}" \
SERVER_GPU="${SERVER_GPU}" \
SIM_GPU="${SIM_GPU}" \
bash "${ROOT}/scripts/run_ar_lownoise_paired_arm.sh" predicted

ROBOTWIN_TEST_NUM=1 \
RUN_DIR="${MEASURED_RUN}" \
PAIR_WITH="${PREDICTED_RUN}" \
HTTP_PORT="${HTTP_PORT}" \
SERVER_GPU="${SERVER_GPU}" \
SIM_GPU="${SIM_GPU}" \
bash "${ROOT}/scripts/run_ar_lownoise_paired_arm.sh" measured

echo "paired_robotwin_smoke: PASS"
echo "predicted=${PREDICTED_RUN}"
echo "measured=${MEASURED_RUN}"
