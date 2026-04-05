#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Resize an existing persistent disk and expand the mounted filesystem on TPU VM.

Usage:
  ./scripts/resize-disk.sh

Configuration is read from environment variables:
  PROJECT_ID    GCP project ID (default: current gcloud project)
  ZONE          TPU zone (default: us-west1-c)
  TPU_NAME      TPU VM name (default: tpu-sprint-machine)
  DISK_NAME     Disk name (default: data-disk)
  NEW_SIZE_GB   New disk size in GB (default: 1500)
  MOUNT_POINT   Mounted path to expand (default: /mnt/disks/data)

Example:
  ZONE=us-central1-a NEW_SIZE_GB=2000 ./scripts/resize-disk.sh
USAGE
}

log() {
  printf '[resize-disk] %s\n' "$*"
}

die() {
  printf '[resize-disk] ERROR: %s\n' "$*" >&2
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
NEW_SIZE_GB="${NEW_SIZE_GB:-1500}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/disks/data}"

[ -n "${PROJECT_ID}" ] || die "PROJECT_ID is empty. Set PROJECT_ID or run gcloud config set project <id>."
[[ "${NEW_SIZE_GB}" =~ ^[0-9]+$ ]] || die "NEW_SIZE_GB must be an integer value in GB."

log "Using project: ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}"

log "Checking disk and TPU existence..."
gcloud compute disks describe "${DISK_NAME}" --zone="${ZONE}" >/dev/null
gcloud compute tpus tpu-vm describe "${TPU_NAME}" --zone="${ZONE}" >/dev/null

log "Resizing disk ${DISK_NAME} to ${NEW_SIZE_GB}GB in ${ZONE}..."
gcloud compute disks resize "${DISK_NAME}" \
  --size="${NEW_SIZE_GB}" \
  --zone="${ZONE}"

log "Expanding filesystem on ${TPU_NAME}:${MOUNT_POINT}..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --zone="${ZONE}" --command="
set -euo pipefail

echo 'Checking mount point...'
findmnt '${MOUNT_POINT}'

DEVICE=\$(findmnt -n -o SOURCE '${MOUNT_POINT}')
echo \"Mounted device: \$DEVICE\"

echo 'Disk layout before resize:'
lsblk
df -h '${MOUNT_POINT}'

if [[ \"\$DEVICE\" =~ [0-9]$ ]] && [[ ! \"\$DEVICE\" =~ /dev/mapper/ ]]; then
  echo 'Mounted path looks like a partition. Expanding partition first...'

  if ! command -v growpart >/dev/null 2>&1; then
    echo 'growpart not found. Installing required package...'
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y cloud-guest-utils
    elif command -v yum >/dev/null 2>&1; then
      sudo yum install -y cloud-utils-growpart
    else
      echo 'Cannot install growpart automatically on this image.'
      exit 1
    fi
  fi

  if [[ \"\$DEVICE\" =~ ^/dev/nvme[0-9]n[0-9]p[0-9]+$ ]]; then
    DISK_BASE=\$(echo \"\$DEVICE\" | sed -E 's/p[0-9]+$//')
    PART_NUM=\$(echo \"\$DEVICE\" | sed -E 's#^/dev/nvme[0-9]n[0-9]p([0-9]+)$#\1#')
  else
    DISK_BASE=\$(echo \"\$DEVICE\" | sed -E 's/[0-9]+$//')
    PART_NUM=\$(echo \"\$DEVICE\" | sed -E 's#^.*/[^0-9]+([0-9]+)$#\1#')
  fi

  echo \"Base disk: \$DISK_BASE\"
  echo \"Partition number: \$PART_NUM\"
  sudo growpart \"\$DISK_BASE\" \"\$PART_NUM\"
else
  echo 'Mounted path appears to be a whole disk. No partition expansion needed.'
fi

echo 'Growing ext4 filesystem...'
sudo resize2fs \"\$DEVICE\"

echo 'Disk layout after resize:'
lsblk
df -h '${MOUNT_POINT}'
"

log "Done. Disk mounted at ${MOUNT_POINT} now reflects the new size."