#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

docker run --platform=linux/amd64 -i --rm \
  -v "$PWD/nao-says-robot-game:/workspace" \
  -w /workspace \
  naoqi \
  "$@"