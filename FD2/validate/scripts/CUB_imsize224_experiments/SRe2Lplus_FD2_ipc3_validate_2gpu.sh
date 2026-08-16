#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IPC=3
exec bash "$SCRIPT_DIR/SRe2Lplus_FD2_ipc5_validate_2gpu.sh"
