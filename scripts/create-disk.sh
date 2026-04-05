#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Create (if needed), attach, format, and mount a persistent disk on a TPU VM.

Usage:
  ./scripts/create-disk.sh

Configuration is read from environment variables:
  PROJECT_ID   GCP project ID (default: current gcloud project)
  ZONE         TPU zone (default: us-west1-c)
  TPU_NAME     TPU VM name (default: tpu-sprint-machine)
  DISK_NAME    Persistent disk name (default: data-disk)
  DISK_SIZE    Initial disk size (default: 500GB)
  DISK_TYPE    Disk type (default: pd-ssd)
  MOUNT_POINT  Mount point inside TPU VM (default: /mnt/disks/data)
  DEVICE_NAME  Preferred block device path inside VM

Example:
  ZONE=us-central1-a TPU_NAME=earthdial-tpu DISK_SIZE=1000GB ./scripts/create-disk.sh
USAGE
}

log() {
  printf '[create-disk] %s\n' "$*"
}

die() {
  printf '[create-disk] ERROR: %s\n' "$*" >&2
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

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
ZONE="${ZONE:-us-west1-c}"
TPU_NAME="${TPU_NAME:-tpu-sprint-machine}"
DISK_NAME="${DISK_NAME:-data-disk}"
DISK_SIZE="${DISK_SIZE:-1500GB}"
DISK_TYPE="${DISK_TYPE:-pd-ssd}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/disks/data}"
DEVICE_NAME="${DEVICE_NAME:-/dev/disk/by-id/google-${DISK_NAME}}"

[ -n "${PROJECT_ID}" ] || die "PROJECT_ID is empty. Set PROJECT_ID or run gcloud config set project <id>."

log "Using project: ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}"

log "Checking TPU VM exists..."
gcloud compute tpus tpu-vm describe "${TPU_NAME}" --zone="${ZONE}" >/dev/null

log "Ensuring disk ${DISK_NAME} exists in ${ZONE}..."
if ! gcloud compute disks describe "${DISK_NAME}" --zone="${ZONE}" >/dev/null 2>&1; then
  gcloud compute disks create "${DISK_NAME}" \
    --zone="${ZONE}" \
    --size="${DISK_SIZE}" \
    --type="${DISK_TYPE}"
else
  log "Disk already exists; skipping creation."
fi

log "Attaching disk to TPU VM..."
attach_output=""
if ! attach_output="$(gcloud alpha compute tpus tpu-vm attach-disk "${TPU_NAME}" --zone="${ZONE}" --disk="${DISK_NAME}" --mode=read-write 2>&1)"; then
  case "${attach_output}" in
    *"already attached"*|*"ALREADY_EXISTS"*)
      log "Disk already attached; continuing."
      ;;
    *)
      printf '%s\n' "${attach_output}" >&2
      die "Failed to attach disk ${DISK_NAME} to ${TPU_NAME}."
      ;;
  esac
fi

log "Formatting (if needed) and mounting disk inside TPU VM..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --zone="${ZONE}" --command "bash -s -- '${DEVICE_NAME}' '${MOUNT_POINT}'" <<'REMOTE'
set -euo pipefail

DEVICE_NAME="$1"
MOUNT_POINT="$2"

echo "Checking disks..."
lsblk

DEVICE="${DEVICE_NAME}"
if [ ! -b "${DEVICE}" ]; then
  echo "Preferred device not found (${DEVICE}). Attempting auto-detection..."
  DEVICE="$(lsblk -dpno NAME,TYPE | awk '$2 == "disk" {print $1}' | grep -vE '/dev/sda$' | head -n 1)"
fi

[ -n "${DEVICE}" ] || { echo "No candidate disk device found."; exit 1; }
[ -b "${DEVICE}" ] || { echo "Resolved device is not a block device: ${DEVICE}"; exit 1; }

echo "Using device: ${DEVICE}"

if ! sudo blkid "${DEVICE}" >/dev/null 2>&1; then
  echo "Formatting ${DEVICE} as ext4..."
  sudo mkfs.ext4 -F "${DEVICE}"
else
  echo "Device already has a filesystem; skipping format."
fi

sudo mkdir -p "${MOUNT_POINT}"

if ! mountpoint -q "${MOUNT_POINT}"; then
  echo "Mounting ${DEVICE} at ${MOUNT_POINT}..."
  sudo mount "${DEVICE}" "${MOUNT_POINT}"
else
  echo "Mount point already active; skipping mount."
fi

sudo chown "$USER:$USER" "${MOUNT_POINT}"

echo "Disk setup complete."
df -h "${MOUNT_POINT}"
REMOTE

log "Done. Disk is ready at ${MOUNT_POINT}."