#!/usr/bin/env bash

set -euo pipefail

# Configurable values (can be overridden via environment variables)
DEVICE="${DEVICE:-/dev/sdb}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/disks/data}"
OWNER_USER="${OWNER_USER:-${SUDO_USER:-$USER}}"
OWNER_GROUP="${OWNER_GROUP:-${OWNER_USER}}"

log() {
	printf '[mount-disk] %s\n' "$*"
}

die() {
	printf '[mount-disk] ERROR: %s\n' "$*" >&2
	exit 1
}

run_as_root() {
	if [[ "$EUID" -eq 0 ]]; then
		"$@"
	else
		sudo "$@"
	fi
}

if [[ ! -b "$DEVICE" ]]; then
	die "Device $DEVICE does not exist or is not a block device."
fi

log "Ensuring mount point exists at $MOUNT_POINT"
run_as_root mkdir -p "$MOUNT_POINT"

if mountpoint -q "$MOUNT_POINT"; then
	CURRENT_DEVICE="$(findmnt -n -o SOURCE "$MOUNT_POINT")"
	if [[ "$CURRENT_DEVICE" == "$DEVICE" ]]; then
		log "$DEVICE is already mounted at $MOUNT_POINT"
	else
		die "$MOUNT_POINT is already mounted by $CURRENT_DEVICE"
	fi
else
	log "Mounting $DEVICE at $MOUNT_POINT"
	run_as_root mount "$DEVICE" "$MOUNT_POINT"
fi

log "Setting ownership to $OWNER_USER:$OWNER_GROUP"
run_as_root chown "$OWNER_USER:$OWNER_GROUP" "$MOUNT_POINT"

log "Done."
df -h "$MOUNT_POINT"