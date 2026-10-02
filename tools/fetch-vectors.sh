#!/usr/bin/env bash
# Downloads the public VP9 test vectors (bitstreams plus the MD5 of every
# decoded frame) into tests/vectors/. They are data published by the WebM
# project for decoder conformance testing:
#   https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/
# The list in tools/vectors.txt is every vp90-2-*, vp91-2-*, vp92-2-* and
# vp93-2-* stream in that bucket except the long film clips (bbb_, sintel_,
# tos_: 1.8 GB between them). About 34 MB in all.
set -euo pipefail
cd "$(dirname "$0")/.."
base=https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx
mkdir -p tests/vectors
while read -r name; do
  [ -z "$name" ] && continue
  for f in "$name" "$name.md5"; do
    [ -s "tests/vectors/$f" ] || curl -sSfL --retry 3 -o "tests/vectors/$f" "$base/$f"
  done
done < tools/vectors.txt
echo "tests/vectors: $(ls tests/vectors | grep -vc '\.md5$') streams"
