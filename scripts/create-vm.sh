#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

usage() {
  cat <<'USAGE'
Create a TPU VM with zone fallback and bootstrap dependencies.

Usage:
  ./scripts/create-vm.sh

Configuration is read from environment variables:
  PROJECT_ID        GCP project ID (default: current gcloud project)
  ZONE              Primary zone to try first (default: us-central1-a)
  ZONE_CANDIDATES   Comma-separated fallback zones
  TPU_NAME          TPU VM name (default: tpu-sprint-machine)
  ACCELERATOR_TYPE  TPU type (default: v5litepod-8)
  VERSION           TPU runtime image (default: tpu-ubuntu2204-base)
  SA_NAME           Service account name (default: tpu-vm-sa)

Example:
  PROJECT_ID=my-proj ZONE=us-west1-c TPU_NAME=earthdial-tpu ./scripts/create-vm.sh
USAGE
}

log() {
  printf '[create-vm] %s\n' "$*"
}

die() {
  printf '[create-vm] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_command gcloud

# =========================
# User-configurable values
# =========================
export PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
if [ -z "${PROJECT_ID}" ]; then
  die "PROJECT_ID is empty. Run 'gcloud config set project <id>' or set PROJECT_ID explicitly."
fi

export ZONE="${ZONE:-us-west1-c}"
export TPU_NAME="${TPU_NAME:-tpu-sprint-machine}"
export ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v5litepod-8}"
export VERSION="${VERSION:-tpu-ubuntu2204-base}"

# Preferred fallback zones for v5litepod-8 / v5e TPUs. Override if you need a
# different search order or a different accelerator family.
export ZONE_CANDIDATES="${ZONE_CANDIDATES:-us-central1-a,us-south1-a,us-west1-c,us-west4-a,europe-west4-b}"

# Service account settings
export SA_NAME="${SA_NAME:-tpu-vm-sa}"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Optional: choose where uv should be installed in this ephemeral setup
export UV_UNMANAGED_INSTALL="/opt/uv"

# =========================
# gcloud project config
# =========================
log "Using project: ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}"

# Enable required APIs
gcloud services enable tpu.googleapis.com iam.googleapis.com

# Create TPU service identity
gcloud beta services identity create \
  --service=tpu.googleapis.com \
  --project="${PROJECT_ID}" || true

# =========================
# Create service account
# =========================
gcloud iam service-accounts create "${SA_NAME}" \
  --project="${PROJECT_ID}" \
  --display-name="TPU VM Service Account" || true

# Grant common TPU roles
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/tpu.admin"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/monitoring.metricWriter"

# =========================
# Create TPU VM
# =========================
zone_list=""

add_zone() {
  zone="$1"

  [ -n "${zone}" ] || return 0

  case ",${zone_list}," in
    *,"${zone}",*) return 0 ;;
  esac

  if [ -n "${zone_list}" ]; then
    zone_list="${zone_list},${zone}"
  else
    zone_list="${zone}"
  fi
}

add_zone "${ZONE}"

OLD_IFS="${IFS}"
IFS=','
set -- ${ZONE_CANDIDATES}
IFS="${OLD_IFS}"

for candidate_zone in "$@"; do
  add_zone "${candidate_zone}"
done

STARTUP_SCRIPT_FILE="$(mktemp)"
trap 'rm -f "${STARTUP_SCRIPT_FILE}"' EXIT

cat >"${STARTUP_SCRIPT_FILE}" <<'EOF'
#! /bin/bash
set -euxo pipefail

# Log everything from the startup script
exec > >(tee /var/log/startup-script.log | logger -t startup-script) 2>&1

export DEBIAN_FRONTEND=noninteractive
export UV_UNMANAGED_INSTALL="/opt/uv"
export PATH="${UV_UNMANAGED_INSTALL}:${PATH}"

echo "Updating apt and installing base packages..."
apt-get update -y
apt-get install -y curl ca-certificates python3 python3-venv

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${UV_UNMANAGED_INSTALL}" sh

echo "Verifying uv..."
"${UV_UNMANAGED_INSTALL}/uv" --version

echo "Writing environment helper..."
cat >/etc/profile.d/tpu-env.sh <<'PROFILE_EOF'
export UV_UNMANAGED_INSTALL="/opt/uv"
export PATH="/opt/uv:${PATH}"
PROFILE_EOF
chmod 644 /etc/profile.d/tpu-env.sh

echo "Running verification..."
python3 - <<'PY'
import os
import subprocess

print("Python version:", subprocess.check_output(["python3", "--version"], text=True).strip())
print("uv path:", os.environ["UV_UNMANAGED_INSTALL"])
PY

echo "Startup script completed successfully."
EOF

SELECTED_ZONE=""

OLD_IFS="${IFS}"
IFS=','
set -- ${zone_list}
IFS="${OLD_IFS}"

for candidate_zone in "$@"; do
  log "Checking ${ACCELERATOR_TYPE} support in ${candidate_zone}..."

  if ! gcloud compute tpus tpu-vm accelerator-types describe "${ACCELERATOR_TYPE}" \
    --project="${PROJECT_ID}" \
    --zone="${candidate_zone}" >/dev/null 2>&1; then
    log "Skipping ${candidate_zone}: accelerator ${ACCELERATOR_TYPE} is not available there."
    continue
  fi

  log "Attempting to create ${TPU_NAME} in ${candidate_zone}..."
  if create_output="$(
    gcloud compute tpus tpu-vm create "${TPU_NAME}" \
      --project="${PROJECT_ID}" \
      --zone="${candidate_zone}" \
      --accelerator-type="${ACCELERATOR_TYPE}" \
      --version="${VERSION}" \
      --service-account="${SA_EMAIL}" \
      --metadata-from-file=startup-script="${STARTUP_SCRIPT_FILE}" 2>&1
  )"; then
    printf '%s\n' "${create_output}"
    SELECTED_ZONE="${candidate_zone}"
    break
  fi

  printf '%s\n' "${create_output}" >&2

  case "${create_output}" in
    *"There is no more capacity in the zone"*)
      log "No capacity in ${candidate_zone}; trying the next zone."
      continue
      ;;
  esac

  die "TPU creation failed in ${candidate_zone} for a non-capacity reason; stopping."
  exit 1
done

if [ -z "${SELECTED_ZONE}" ]; then
  die "Failed to create ${TPU_NAME}. None of these zones had both support and available capacity: ${zone_list}"
fi

log "TPU ${TPU_NAME} created in zone ${SELECTED_ZONE}."
log "Use this zone for disk steps: export ZONE=${SELECTED_ZONE}"
