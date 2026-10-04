#!/usr/bin/env bash
# End-to-end decode / encode throughput at 720p and 1080p, single-threaded
# and on all cores. The source is the 1080p test vector (fetch-vectors.sh)
# decoded to raw video, and cropped to 1280x720; the decode inputs are that
# vector itself (libvpx, one tile column) and this crate's own encodes of
# the sources (four tile columns).
#
#   tools/bench.sh [WORKDIR] [RUNS]
#
# WORKDIR holds the raw sources (about 1 GB) and the encodes; default
# target/bench. Each figure is the best of RUNS (default 5).
# VP9_FORCE_SCALAR=1 measures the scalar kernels.
set -euo pipefail
cd "$(dirname "$0")/.."
work=${1:-target/bench}
runs=${2:-5}
mkdir -p "$work"
cargo build --release --example vp9_bench
b=target/release/examples/vp9_bench
vec=tests/vectors/vp90-2-02-size-lf-1920x1080.webm
[ -s "$work/src1080.yuv" ] || $b yuv "$vec" "$work/src1080.yuv"
[ -s "$work/src720.yuv" ] || $b yuv "$vec" "$work/src720.yuv" --crop 1280x720
[ -s "$work/enc1080.ivf" ] || $b enc "$work/src1080.yuv" 1920 1080 --frames 60 --runs 1 --out "$work/enc1080.ivf"
[ -s "$work/enc720.ivf" ] || $b enc "$work/src720.yuv" 1280 720 --frames 60 --runs 1 --out "$work/enc720.ivf"
for t in 1 0; do
  $b dec "$vec" --threads $t --runs "$runs"
  $b dec "$work/enc1080.ivf" --threads $t --runs "$runs"
  $b dec "$work/enc720.ivf" --threads $t --runs "$runs"
  $b enc "$work/src1080.yuv" 1920 1080 --frames 10 --threads $t --runs 2
  $b enc "$work/src720.yuv" 1280 720 --frames 10 --threads $t --runs 2
done
