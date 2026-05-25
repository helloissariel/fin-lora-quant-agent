#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
${PY:-python} data/prepare_data.py "$@"
