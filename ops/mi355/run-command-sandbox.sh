#!/usr/bin/env bash
# Run one trusted manifest command against an approved read-write worktree.
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 WORKSPACE OUTPUT_DIR GPU COMMAND" >&2
  exit 2
fi

workspace="$(realpath "$1")"
output="$(realpath "$2")"
gpu="$3"
command="$4"
image="${ROCM_WIKI_MI355_IMAGE:-}"

if [[ -z "${image}" || "${image}" != *@sha256:* ]]; then
  echo "ROCM_WIKI_MI355_IMAGE must be pinned by sha256 digest" >&2
  exit 2
fi
if command -v podman >/dev/null 2>&1; then
  runtime=podman
elif command -v docker >/dev/null 2>&1; then
  runtime=docker
else
  echo "podman or docker is required" >&2
  exit 2
fi

mkdir -p "${output}"
caller_uid="$(id -u)"
caller_gid="$(id -g)"
device_args=(--device=/dev/kfd)
device_paths=(/dev/kfd)
if [[ -n "${ROCM_WIKI_DRI_DEVICE:-}" ]]; then
  device_args+=("--device=${ROCM_WIKI_DRI_DEVICE}")
  device_paths+=("${ROCM_WIKI_DRI_DEVICE}")
else
  device_args+=(--device=/dev/dri)
  device_paths+=(/dev/dri)
fi

group_args=()
declare -A added_groups=()
for device in "${device_paths[@]}"; do
  device_gid="$(stat -c '%g' "${device}")"
  if [[ -z "${added_groups[${device_gid}]+x}" ]]; then
    group_args+=("--group-add=${device_gid}")
    added_groups["${device_gid}"]=1
  fi
done

exec "${runtime}" run --rm \
  --pull=never \
  --network=none \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --user="${caller_uid}:${caller_gid}" \
  --pids-limit=1024 \
  --ipc=private \
  --shm-size=1g \
  --tmpfs=/tmp:rw,exec,nosuid,nodev,size=4g \
  "${device_args[@]}" \
  "${group_args[@]}" \
  -e "HOME=/tmp" \
  -e "ROCR_VISIBLE_DEVICES=${gpu}" \
  -e "HIP_VISIBLE_DEVICES=${gpu}" \
  -e "CUDA_VISIBLE_DEVICES=${gpu}" \
  -v "${workspace}:/workspace:rw" \
  -v "${output}:/evidence:rw" \
  -w /workspace \
  "${image}" \
  bash -lc "${command}"
