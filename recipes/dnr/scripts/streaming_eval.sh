#! /bin/bash

# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

set -euo pipefail

model_path=$1
chunk_frames=${2:-8}
device=${3:-cuda}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
"${repo_root}/tools/online/run_streaming_eval.sh" dnr "${model_path}" "${chunk_frames}" "${device}"
