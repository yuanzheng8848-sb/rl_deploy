#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_SERL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPENARM_CAN_ROOT="${RL_SERL_ROOT}/third_party/openarm_can"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <can_interface> [options passed to configure_socketcan.sh]"
  echo "Example: $0 can0"
  echo "Example: $0 can0 -fd"
  exit 1
fi

exec "${OPENARM_CAN_ROOT}/setup/configure_socketcan.sh" "$@"
