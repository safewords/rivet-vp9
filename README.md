# rivet-vp9

[![CI](https://github.com/rivet-transcoder/rivet-vp9/actions/workflows/ci.yml/badge.svg)](https://github.com/rivet-transcoder/rivet-vp9/actions/workflows/ci.yml)

A **VP9** decoder and encoder in Rust: no C, no system libraries, no build
script, nothing to install on a build host. Written from the *VP9 Bitstream
& Decoding Process Specification* (v0.6 / v0.7, Google and Argon Design),
not translated from any other implementation. The decoder is bit-exact on
**352 of the 353** public VP9 test vectors it was run on (the numbers are
[below](#how-it-is-checked)); the encoder writes profile 0 key and inter
frames that decode to what it reconstructed, sample for sample.

Written for the **[rivet](https://github.com/rivet-transcoder/rivet)**
transcoder, as its VP9 codec on both sides. Usable on its own by anything
that has VP9 frames (from IVF, WebM / Matroska, MP4) and wants planar
pictures back, or planar pictures and wants VP9.

This is the first milestone of a longer effort: the decoder is complete
but single-threaded and scalar; the encoder is deliberately simple. What is
and is not there is listed precisely below.

Published as `rivet-vp9`; **imported as `vp9`** (`use vp9::…`). One
dependency (`thiserror`), no features, no build script.

```toml
[dependencies]
vp9 = { package = "rivet-vp9", git = "https://github.com/rivet-transcoder/rivet-vp9", branch = "develop" }
```

## What it decodes

Everything in the specification:

| | |
|---|---|
| **Profiles** | 0, 1, 2 and 3: 8-, 10- and 12-bit; 4:2:0, 4:2:2, 4:4:0 and 4:4:4; every colour space (sRGB, planes G, B, R, decodes like the others; no vector carries it) |
| **Frames** | key frames, inter frames, intra-only frames, hidden frames, `show_existing_frame`, error-resilient and frame-parallel modes, frame size changes, superframes (Annex B) |
| **Syntax** | the uncompressed and compressed headers, forward probability updates and backward adaptation over four frame contexts, all partitions down to 4x4, segmentation (map, temporal prediction, quantiser / loop filter / reference / skip features), tiles (columns and rows), lossless mode |
| **Prediction** | all ten intra predictors at 4x4 to 32x32; inter prediction from three references, single and compound, the regular / smooth / sharp 8-tap and bilinear filters, switchable per block, eighth-sample motion vectors, the full motion vector prediction process including the previous frame's vectors, **scaled references** (a reference of another size, 2:1 down to 1:16) |
| **Residual** | coefficient tokens at every transform size, DCT / ADST in each combination, the 32x32 DCT, the Walsh-Hadamard transform, dequantisation at all bit depths |
| **Loop filter** | the 4-, 8- and 16-wide filters, level by segment, reference and mode deltas, sharpness |

Output is a [`Frame`](src/frame.rs): the visible picture, planes Y, U, V
packed one after another, one byte per sample at 8 bits and little-endian
`u16` above — the shape of rivet-h26x's `Picture`, and the layout the test
vectors' MD5s hash. `Decoder::decode(&packet)` takes one container packet
(a frame or a superframe) and returns the frame it shows, if any; a
superframe that shows several frames returns the last, as the reference
decoder outputs (`Decoder::decode_all` returns every one).

Not there yet:

- **Speed.** Decoding is single-threaded scalar Rust: about 570 frames/s
  at 426x240, 170 at 854x356, 23 at 1920x1080 and 8 at 3840x2160 on one
  core of the machine it was written on. No tile or frame threading, no
  SIMD, no frame-buffer pool (each frame allocates; output copies).
- **Error recovery.** A corrupt frame returns `Error::Bitstream` and is
  dropped; there is no concealment. Malformed input never panics (property
  tests below), and frames larger than `DEFAULT_MAX_PIXELS` (8192x8192,
  adjustable) are refused before allocation.
- Conformance requirements that do not change the output (padding bit
  values, the 2:1 / 1:16 scaling bound on references the frame does not
  use) are not all checked.

## What it encodes

Profile 0 (8-bit 4:2:0), one packet per frame, any size from 1x1 up:

- **Key frames**: every block coded with the best of the ten intra modes
  (luma and chroma chosen separately by rate-distortion cost on the actual
  reconstruction).
- **Inter frames**: single-reference prediction from the previous frame —
  whole-pixel diamond search then half- and quarter-sample refinement with
  the decoder's 8-tap filter; NEARESTMV, NEARMV, ZEROMV or NEWMV (motion
  vectors coded against the specification's motion vector prediction);
  intra as the alternative per block.
- **Fixed quantiser** (`Config::quantizer`, the `base_q_idx` 0–255) with a
  dead-zone rounding quantiser; **quantiser 0 is lossless** (the 4x4
  Walsh-Hadamard transform, exact).
- **Fixed partition**: square blocks of `Config::block_size` (8, 16, 32 or
  64), split or halved where the frame edge forces it, each with the
  largest transform that fits; DCT and ADST as the intra mode implies.
- **Coefficient probability updates**: a first pass over the frame
  counts how each coefficient context's first three probabilities were
  used, the compressed header sends the new probabilities that save more
  than their update costs (the specification's `diff_update_prob` code),
  and a second pass codes the frame with them, replaying the first pass's
  block decisions. Between 2% and 9% smaller than the default probabilities
  on the content below.
- A loop filter level derived from the quantiser (or set).
- A boolean encoder that is the exact inverse of the decoder's, and the
  minimum number of tile columns the frame width requires.

The encoder reconstructs through the decoder's own prediction, motion
vector prediction and inverse-transform code, and keeps its reference
frames by decoding its own packets: what it predicts from is what a decoder
has. Debug builds also compare its reconstruction with the decoded packet
sample by sample.

Not there yet (in rough order of value):

- **Rate control** — only a fixed quantiser; no target bitrate, two-pass,
  or adaptive quantisation (segmentation).
- **Partition and transform-size search** — the partition is fixed; no
  `TX_MODE_SELECT`, no sub-8x8 blocks.
- **More references** — only LAST; no golden / alt-ref frames, hidden
  frames, compound prediction, or reference scaling.
- **Probability updates** — every frame codes with the default
  probabilities (no forward updates, and inter frames are error resilient
  so there is no backward adaptation either); no high-precision (1/8)
  motion vectors, no switchable interpolation filters.
- **Profiles 1–3.** What they need: the profile bits and colour config
  (bit depth and subsampling fields) in the header; source padding, the
  plane geometry and the distortion sums generalised from the hard-coded
  4:2:0 shifts (the reconstruction engine — the decoder's — already handles
  every format); for 4:2:2 and 4:4:0, falling back to a split where a forced
  edge partition gives a chroma block size the specification forbids; for
  10 / 12-bit, the high-bit-depth quantiser tables, the extra `high_bit`s of
  category-6 tokens, and the rate-distortion multiplier scaled by
  2^(2(bitdepth - 8)). Samples are already `u16` throughout.
- Speed: about 15 frames/s at 352x288, single-threaded.

Measured on the decoded frames of `vp90-2-03-size-226x226.webm` (10 frames
of natural video, `tests/encode.rs`), 16x16 blocks, a key frame then nine
inter frames:

| quantiser | bytes (10 frames) | key frame | per inter frame | PSNR Y | PSNR U | PSNR V |
|---|---|---|---|---|---|---|
| 16 | 144 251 | 25 550 | 13 189 | 49.87 dB | 51.54 dB | 51.24 dB |
| 48 | 81 024 | 16 739 | 7 143 | 43.48 dB | 47.54 dB | 47.15 dB |
| 96 | 50 712 | 11 431 | 4 365 | 38.84 dB | 44.69 dB | 44.21 dB |
| 160 | 20 738 | 5 097 | 1 738 | 31.56 dB | 39.21 dB | 38.70 dB |
| 240 | 4 520 | 847 | 408 | 22.63 dB | 32.67 dB | 32.49 dB |
| 0 (lossless) | 268 262 | | | exact | exact | exact |

At quantiser 64 the ten frames take 67 631 bytes with inter frames and
134 304 coded all-intra.

## How it is checked

- **The WebM project's VP9 test vectors**: bitstreams published with the
  MD5 of every frame the reference decoder shows. `tools/fetch-vectors.sh`
  downloads the 353 `vp90-2-*`, `vp91-2-*`, `vp92-2-*` and `vp93-2-*`
  streams (about 34 MB; the film clips `bbb_`, `sintel_`, `tos_`, 1.8 GB and
  without MD5s, are left out) and `tests/vectors.rs` decodes each and
  compares every shown frame. **352 of 353 pass**, every frame bit-exact:

  | group | what it exercises | pass |
  |---|---|---|
  | vp90-2-00 | quantiser 0–63 (lossless included) | 64 / 64 |
  | vp90-2-01 | loop filter sharpness | 7 / 7 |
  | vp90-2-02, vp90-2-03 | frame sizes 8x8 to 66x66, 130 to 226, odd sizes, 1920x1080; in-stream resizing at encode speeds 5 and 7; delta quantisers | 164 / 164 |
  | vp90-2-05, 06, 13, 14, 18, 21 | resizing and scaled references, bilinear | 69 / 69 |
  | vp90-2-07, 08 | frame-parallel mode, tiles 1x2 to 4x4 and 1x8 (up to 3840x2160) | 12 / 12 |
  | vp90-2-09, 11, 12, 15, 19 | aq, loop filter deltas, subpixel, odd sizes, droppable frames, segmentation keys, a fuzzed file, skip | 15 / 15 |
  | vp90-2-10, 16, 17 | show-existing-frame, **intra-only frames** | 4 / 4 |
  | vp90-2-20, 22 | superframes (big indexes, hidden frames), spatial SVC | 6 / 6 |
  | vp91-2-04 | profile 1: 4:2:2, 4:4:0, 4:4:4 | 3 / 4 |
  | vp92-2-20, vp93-2-20 | profiles 2 and 3: 10 / 12-bit, 4:2:0 to 4:4:4 | 8 / 8 |

  The one failure is `vp91-2-04-yv444.webm`: its MD5 file is in an older
  format than every other vector's and its key frame does not parse to the
  end of its tile data here while its inter frames do — consistent with a
  stream from before the profile 1 syntax was final; `vp91-2-04-yuv444.webm`
  passes. **Intra-only frames**: `vp90-2-16-intra-only.webm` passes, and
  the first frame (a key frame) of 352 of the 353 vectors is bit-exact.
  Fourteen small vectors are committed in [`tests/data`](tests/data/README.md)
  so `cargo test` checks real streams without the download.
- **The encoder** (`tests/encode.rs`): every packet decodes; lossless is
  exact at every size from 1x1 and every block size; PSNR falls and size
  falls as the quantiser rises; inter frames cost less than intra; and
  debug builds check the encoder's reconstruction against the decoder's
  output sample by sample.
- **Malformed input** (`tests/fuzz.rs`, proptest): arbitrary bytes, and the
  committed vectors with bits flipped, bytes cut and garbage spliced in,
  decoded in debug builds (overflow checks on) — errors, never a panic.
- **Units against the specification**: the boolean decoder against a
  bit-by-bit transcription of 9.2 and the encoder's inverse; every inverse
  DCT and ADST against the real transforms they approximate (which checks
  the butterfly transcription); the forward transforms round-tripping
  through the decoder's inverse; scans are permutations; the subpixel
  filters sum to 128.

## Provenance and licensing

Written from the specification's text; **no VP9 implementation's source was
read** — not libvpx, not FFmpeg's libavcodec, not any other — and none was
run: the tests use only the specification-derived checks above, round trips
through this crate, and the published test-vector bitstreams with their
MD5s (data, downloaded from the WebM project's public test-data bucket; the
committed ones are listed with their source in
[`tests/data/README.md`](tests/data/README.md)). The specification's tables
(probabilities, scans, quantiser lookups, filter taps) are data and were
transcribed by [`tools/gen_tables.py`](tools/gen_tables.py) from the
published PDF, which regenerates [`src/tables.rs`](src/tables.rs) byte for
byte from either v0.6 or v0.7. See [NOTICE](NOTICE).

**Patents.** Google grants a royalty-free patent licence for VP9 with the
WebM project; others have asserted patents on VP9 too. This licence grants
no rights under any patent, and the authors make no claim about whether a
licence is needed for any use.

Licensed under the Open Encoding Attribution License 1.0
([LICENSE.md](LICENSE.md)): source-available, not open source. §4 requires
the [NOTICE](NOTICE) to travel with any distribution; §5 requires a short
credit for commercial use that reaches third parties.

## Using it

```rust
// Decoding: one container packet at a time.
let mut dec = vp9::Decoder::new();
for packet in packets {
    if let Some(frame) = dec.decode(&packet)? {
        // frame.width, frame.height, frame.bit_depth, frame.chroma
        // frame.plane(0), frame.plane(1), frame.plane(2)
    }
}

// Encoding: 8-bit 4:2:0 frames in, one packet each out.
let mut cfg = vp9::Config::new(1280, 720);
cfg.quantizer = 60;
let mut enc = vp9::Encoder::new(cfg);
let packet = enc.encode(&frame)?;
```

`vp9::ivf` reads and writes IVF; `vp9::superframe` splits and builds
superframes. The examples decode IVF to raw planar video with per-frame
MD5s (`cargo run --release --example ivfdec -- in.ivf out.yuv`) and encode
raw I420 to IVF (`--example ivfenc -- in.yuv 352 288 out.ivf 60`).

## Specification notes

Points where the specification text needed interpretation; each was
settled by the test vectors:

- **The more_coefs counting "special case"** (9.3.4) is announced but its
  description is missing from the text of both v0.6 and v0.7. Plain
  counting of every decoded `more_coefs` value, like any other element,
  matches every vector.
- **Partition probabilities** (9.3.2): the text says `kf_partition_probs`
  when FrameIsIntra is 0 — inverted; key and intra-only frames use them.
- **Superframe frame sizes** (B.2) are written `f(SzBytes)` but are
  little-endian byte counts.
- **Loop filter levels** (8.8.1 step 3) test `loop_filter_delta_update`
  where `loop_filter_delta_enabled` is meant (the two agree whenever the
  update flag is read); 8.8.4 indexes LvlLookup with `segment_id` where its
  own variable is `segment`.
- **The inverse DCT** (8.7.1.3) loop bound "2n-7" in step 5b is 2·n − 7,
  confirmed by checking the transform against the real DCT.
- **Above-right intra edges** (8.5.1) are used only for 4x4 transform
  blocks not on the block's right edge — taken literally, and right.
- **Several shown frames in one superframe** (Annex B allows it): the
  vectors expect the last one per packet (`vp90-2-22-svc_1280x720_3.ivf`).
- **Inferred values are counted** (9.3): a `partition` forced to SPLIT at
  the frame edge, and the implied `mv_hp` / `mv_class0_hp` of 1, are
  counted for adaptation like decoded ones.
