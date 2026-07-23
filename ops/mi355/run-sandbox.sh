#!/usr/bin/env bash
# Execute the trusted validation controller against an approved, read-only
# candidate payload. No GitHub credentials or network are exposed.
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 CONTROLLER PAYLOAD_DIR CANDIDATE_REPO OUTPUT_DIR GPU" >&2
  exit 2
fi

controller="$(realpath "$1")"
payload="$(realpath "$2")"
candidate="$(realpath "$3")"
output="$(realpath "$4")"
gpu="$5"

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
  -e "ROCM_WIKI_VALIDATION_PAYLOAD=/payload" \
  -e "ROCM_WIKI_REPOSITORY_DIR=/candidate" \
  -v "${controller}:/control/run.py:ro" \
  -v "${payload}:/payload:ro" \
  -v "${candidate}:/candidate:ro" \
  -v "${output}:/evidence:rw" \
  "${image}" \
  python3 /control/run.py --output /evidence/validation --timeout 120
