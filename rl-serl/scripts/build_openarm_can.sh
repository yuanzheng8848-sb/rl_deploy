#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_SERL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OPENARM_CAN_ROOT="${RL_SERL_ROOT}/third_party/openarm_can"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "openarm_can must be built inside Linux/WSL because it uses SocketCAN."
  exit 1
fi

if [[ ! -d "${OPENARM_CAN_ROOT}" ]]; then
  echo "Missing ${OPENARM_CAN_ROOT}"
  exit 1
fi

command -v cmake >/dev/null || { echo "Missing cmake"; exit 1; }
command -v python >/dev/null || { echo "Missing python"; exit 1; }

if ! command -v ninja >/dev/null; then
  echo "Missing ninja. Install it with: sudo apt install ninja-build"
  exit 1
fi

echo "Building C++ openarm_can library..."
cmake \
  -S "${OPENARM_CAN_ROOT}" \
  -B "${OPENARM_CAN_ROOT}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -GNinja
cmake --build "${OPENARM_CAN_ROOT}/build"

echo "Installing C++ openarm_can library. This may ask for sudo."
sudo cmake --install "${OPENARM_CAN_ROOT}/build"

echo "Installing Python openarm_can binding into the active Python environment..."
python -m pip install "${OPENARM_CAN_ROOT}/python"

echo "openarm_can build/install complete."
python -c "import openarm_can as oa; print('openarm_can import OK:', oa.MotorType.DM4310)"
