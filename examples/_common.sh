#!/usr/bin/env bash
# Shared build/run boundary. CI may set ROCM_WIKI_BUILD_ONLY=1 to compile and
# inspect ISA without executing newly built GPU or host code.

rocm_wiki_run() {
  if [[ "${ROCM_WIKI_BUILD_ONLY:-0}" == "1" ]]; then
    echo "build-only: skipped execution of $*"
    return 0
  fi
  "$@"
}
