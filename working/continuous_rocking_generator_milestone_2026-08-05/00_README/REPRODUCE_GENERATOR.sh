#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../04_CONTINUOUS_GENERATOR"

python3.13 test_support_stride_generator_audit.py   --input ../03_STRIDE_INVOLUTION/support_stride_involution_audit_raw.npz   2>&1 | tee reproduced_support_stride_generator_audit_console.log
