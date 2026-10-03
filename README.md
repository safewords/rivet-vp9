# rivet-vp9

[![CI](https://github.com/rivet-transcoder/rivet-vp9/actions/workflows/ci.yml/badge.svg)](https://github.com/rivet-transcoder/rivet-vp9/actions/workflows/ci.yml)

A **VP9** decoder and encoder in Rust: no C, no system libraries, no build
script, nothing to install on a build host. Written from the *VP9 Bitstream
& Decoding Process Specification* (v0.6 / v0.7, Google and Argon Design),
not translated from any other implementation. The decoder is bit-exact on
**all 353** public VP9 test vectors it was run on (the numbers are
[below](#how-it-is-checked)); the encoder writes all four profiles (8 to
12 bits, 4:2:0 to 4:4:4) with a rate-distortion partition and transform
search and a target bitrate (one or two passes), and its frames decode to
what it reconstructed, sample for sample.

Written for the **[rivet](https://github.com/rivet-transcoder/rivet)**
transcoder, as its VP9 codec on both sides. Usable on its own by anything
that has VP9 frames (from IVF, WebM / Matroska, MP4) and wants planar
pictures back, or planar pictures and wants VP9.

This is an early milestone of a longer effort: the decoder is complete but
single-threaded and scalar; the encoder is complete enough to use but slow
and missing several tools. What is and is not there is listed precisely
below.

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
- **Error recovery.** A corrupt frame returns `Error::Bitstream`; there is
  no concealment, and the decoder's state after an error is whatever the
  failed frame changed before it failed, so decoding should resume at the
  next key frame. Malformed input has not made it panic under the property
  tests below, and frames larger than `DEFAULT_MAX_PIXELS` (8192x8192,
  adjustable) are refused before allocation.
- Conformance requirements that do not change the output (padding bit
  values, the 2:1 / 1:16 scaling bound on references the frame does not
  use) are not all checked.

## What it encodes

Profiles 0 to 3 — 8, 10 and 12 bits; 4:2:0, 4:2:2, 4:4:0 and 4:4:4 (and
sRGB, coded 4:4:4) — one packet per frame, any size from 1x1 up. The
profile follows from `Config::bit_depth` and `Config::chroma`
(`Config::profile()`).

- **Key frames**: every block coded with the best of the ten intra modes.
- **Inter frames**: single-reference prediction from **LAST** (the previous
  frame) or **GOLDEN** (the last key frame, replaced every
  `Config::golden_interval` frames, default 8, by a frame coded finer: at
  3/4 of the quantiser, or with twice an inter frame's budget under rate
  control) — whole-pixel diamond search then half- and quarter-sample
  refinement with the decoder's 8-tap filter, per reference; NEARESTMV,
  NEARMV, ZEROMV or NEWMV (motion vectors coded against the specification's
  motion vector prediction); intra as the alternative per block.
- **Rate-distortion search** (`Config::speed`): every candidate — a
  partition of a region, a transform size, intra against inter — is coded
  for real (prediction, transform, quantisation, reconstruction) and its
  syntax written to a bit counter, which gives its exact rate under the
  frame's probabilities; the cheapest by squared error + λ·rate wins (λ
  from the quantiser step, so scaled by 2^(2(bitdepth − 8)) at high bit
  depth). A superblock's state is saved and restored around each trial and
  the winning decisions replayed into the boolean encoder.
  - speed 0: the partition from 64x64 down to 8x8 with NONE, HORZ, VERT and
    SPLIT; every transform size (`TX_MODE_SELECT`).
  - speed 1 (default): NONE and SPLIT (HORZ / VERT where the frame edge
    forces them); the two largest transform sizes.
  - speed 2: the fixed partition of `Config::block_size` (8, 16, 32 or 64),
    the largest transform that fits — the encoder before this release.

  In 4:2:2 and 4:4:0, partitions whose chroma block the specification
  forbids are never chosen (a forced edge HORZ / VERT becomes a SPLIT).
- **Rate control**: a fixed quantiser (`Config::quantizer`, the
  `base_q_idx` 0–255; **0 is lossless**, the 4x4 Walsh-Hadamard transform),
  or a **target bitrate** (`Config::target_bitrate` at
  `Config::frame_rate`). Each frame gets a budget and is coded at the
  quantiser a per-frame-type model (`bits = c · qstep^-s`) predicts, then
  recoded — up to `Config::max_recodes` times, default 2 — while it misses
  by more than 12%; the attempt nearest the budget is sent. One pass: the
  bitrate's share per frame (key frames 4x, golden frames 2x, budget-neutral
  over the golden interval), less the overspend so far spread over the next
  second. **Two passes**: a `FirstPass` codes the clip at one quantiser;
  with its statistics in `Config::two_pass` the clip's remaining budget is
  divided by first-pass size^0.8 (key and golden frames weighted as above),
  and each frame's search starts from its own first-pass complexity.
- **Coefficient probability updates**: a first pass over the frame
  counts how each coefficient context's first three probabilities were
  used, the compressed header sends the new probabilities that save more
  than their update costs (the specification's `diff_update_prob` code),
  and a second pass codes the frame with them, replaying the first pass's
  decisions.
- Up to 12-bit: the high-bit-depth quantiser tables, and the extra high
  bits of category-6 tokens (coefficients past 16 450).
- A loop filter level derived from the quantiser (or set).
- A boolean encoder that is the exact inverse of the decoder's, and the
  minimum number of tile columns the frame width requires.

The encoder reconstructs through the decoder's own prediction, motion
vector prediction and inverse-transform code, and keeps its reference
frames by decoding its own packets: what it predicts from is what a decoder
has, and `Encoder::reconstruction()` returns it (the last frame, exactly as
a decoder outputs it). Debug builds also loop filter the encoder's own
reconstruction and compare it with the decoded packet sample by sample.
`Encoder::force_keyframe()`, `last_quantizer()` and `last_was_keyframe()`
complete the API.

Not there yet (in rough order of value):

- **Speed.** Single-threaded scalar Rust: at 352x288, about 1.7 frames/s at the default speed 1, 0.6 at speed 0
  and 10 at speed 2 (1.2 / 0.6 / 12 with LAST only).
- **Sub-8x8 blocks** (4x4, 4x8, 8x4 partitions of an 8x8), and the
  transform size is searched for the modes chosen at the largest one rather
  than jointly.
- **ALTREF, hidden frames and compound prediction** — compound prediction
  needs two references on opposite sides in time (different sign biases),
  i.e. a frame coded ahead of its display time and not shown; there is no
  lookahead. No reference scaling.
- **Other probability updates** — only the coefficient probabilities are
  updated; mode, partition, skip and motion vector probabilities stay at
  their defaults, and inter frames are error resilient, so there is no
  backward adaptation and no motion vectors from the previous frame. No
  high-precision (1/8) motion vectors, no switchable interpolation filters.
- **Adaptive quantisation** (segmentation), and rate control below the
  frame (a frame's quantiser is uniform). No buffer model (VBV / CBR
  constraints) beyond the overspend correction.
- A first pass that is a full encode at one quantiser (it doubles the
  time), not a cheap analysis.

### Measurements

Measured with `tests/encode.rs` (`cargo test --release --test encode --
--ignored --nocapture` for the long ones; the 352x288 clips need
`tools/fetch-vectors.sh`), on the decoded frames of test vectors,
single-threaded.

**What each tool saves**, as BD-rate (the average bitrate difference at
equal luma PSNR, over quantisers 40, 80, 120, 160 and 200) against the
fixed 16x16 partition with LAST only:

| | 226x226, 10 frames | 352x288, 40 frames (`droppable_1`) | 352x288 frames/s |
|---|---|---|---|
| fixed 32x32 partition (speed 2) | +36.0% | +17.0% | 6.2 |
| transform-size search alone (16x16 partition) | −14.1% | −3.8% ¹ | 7.8 ¹ |
| partition search alone (largest transforms) | −24.8% | −25.7% ¹ | 2.6 ¹ |
| speed 1, LAST only | −33.1% | −24.6% | 1.2 |
| speed 0, LAST only | −36.1% | −27.2% | 0.6 |
| speed 1 + GOLDEN every 8, coded finer (**the default**) | −30.2% | −33.4% | 1.7 |
| speed 0 + GOLDEN every 8 | −33.1% | −36.0% | 0.6 |
| speed 2 (16x16) + GOLDEN every 8 | +3.7% | −7.5% | 10.3 |

¹ Measured on the first 20 frames, with temporary settings that are not
`Config` options. GOLDEN costs on the 10-frame clip — the finer golden
frame (frame 8) has only one frame after it to pay off — and gains
11.7% at speed 1 over 40 frames (2.9% of it from the reference, the rest
from coding the golden frame finer).

**Fixed quantiser**, the default settings, `vp90-2-03-size-226x226.webm`
(10 frames, a key frame then nine inter frames); the previous release
(fixed 16x16 partition, LAST only) for comparison:

| quantiser | bytes | key frame | per inter frame | PSNR Y | PSNR U | PSNR V | previous release |
|---|---|---|---|---|---|---|---|
| 16 | 109 548 | 18 592 | 10 106 | 50.64 dB | 51.81 dB | 51.46 dB | 141 491 bytes, 49.87 dB |
| 48 | 57 591 | 12 094 | 5 055 | 44.73 dB | 48.15 dB | 47.67 dB | 79 581 bytes, 43.48 dB |
| 96 | 36 158 | 9 655 | 2 945 | 40.35 dB | 45.81 dB | 45.21 dB | 49 031 bytes, 38.84 dB |
| 160 | 17 026 | 5 087 | 1 327 | 33.33 dB | 40.44 dB | 39.78 dB | 18 921 bytes, 31.56 dB |
| 240 | 5 465 | 731 | 526 | 23.64 dB | 33.49 dB | 33.01 dB | 4 337 bytes, 22.63 dB |
| 0 (lossless) | 227 690 | | | exact | exact | exact | 245 092 bytes |

At quantiser 64 the ten frames take 47 392 bytes with inter frames and
117 720 coded all-intra.

**Every profile**: the first four frames of the same clip converted to
each format (chroma resampled by nearest neighbour; 10 and 12 bits by
widening the 2x2 mean of the 8-bit samples, so the low bits carry detail),
bytes for four frames and mean PSNR against the format's peak:

| | q 32 | q 96 | q 192 |
|---|---|---|---|
| 8-bit 4:2:0 (profile 0) | 27 627 B, Y 47.18 / U 50.01 / V 49.65 dB | 14 013 B, 40.35 / 45.36 / 44.93 | 3 599 B, 28.97 / 36.33 / 35.76 |
| 8-bit 4:2:2 (profile 1) | 30 872 B, 47.18 / 50.19 / 49.82 | 15 031 B, 40.38 / 45.79 / 45.45 | 3 816 B, 29.03 / 38.13 / 37.58 |
| 8-bit 4:4:0 (profile 1) | 30 975 B, 47.18 / 50.24 / 49.81 | 14 965 B, 40.36 / 45.96 / 45.45 | 3 738 B, 29.03 / 37.46 / 37.73 |
| 8-bit 4:4:4 (profile 1) | 36 088 B, 47.09 / 50.62 / 50.12 | 16 462 B, 40.31 / 46.31 / 45.79 | 3 917 B, 28.98 / 39.34 / 38.80 |
| 10-bit 4:2:0 (profile 2) | 30 987 B, 49.63 / 52.85 / 52.42 | 14 033 B, 40.62 / 45.91 / 45.46 | 3 615 B, 29.07 / 36.27 / 36.01 |
| 10-bit 4:2:2 (profile 3) | 34 752 B, 49.60 / 52.88 / 52.38 | 14 985 B, 40.61 / 46.42 / 46.00 | 3 810 B, 29.02 / 37.95 / 37.49 |
| 10-bit 4:4:0 (profile 3) | 34 985 B, 49.59 / 52.89 / 52.29 | 14 983 B, 40.59 / 46.61 / 45.94 | 3 761 B, 29.05 / 37.33 / 37.59 |
| 10-bit 4:4:4 (profile 3) | 40 933 B, 49.56 / 53.45 / 52.82 | 16 417 B, 40.60 / 46.94 / 46.41 | 3 929 B, 29.01 / 39.45 / 38.79 |
| 12-bit 4:2:0 (profile 2) | 32 107 B, 50.17 / 53.42 / 52.97 | 14 030 B, 40.69 / 46.06 / 45.47 | 3 595 B, 29.06 / 36.43 / 35.93 |
| 12-bit 4:2:2 (profile 3) | 36 112 B, 50.16 / 53.40 / 52.86 | 15 133 B, 40.67 / 46.41 / 46.00 | 3 803 B, 29.03 / 37.78 / 37.48 |
| 12-bit 4:4:0 (profile 3) | 36 522 B, 50.15 / 53.45 / 52.83 | 15 042 B, 40.67 / 46.77 / 46.04 | 3 798 B, 29.03 / 37.49 / 37.54 |
| 12-bit 4:4:4 (profile 3) | 42 740 B, 50.12 / 53.99 / 53.35 | 16 480 B, 40.65 / 47.00 / 46.50 | 3 933 B, 29.04 / 39.46 / 38.74 |

**Rate control**, 30 frames/s, the achieved bitrate over the whole clip
and its mean luma PSNR:

| clip | target | one pass | two passes |
|---|---|---|---|
| `vp90-2-09-aq2.webm`, 352x240, 100 frames | 150 kb/s | 160.6 kb/s (+7.1%), 22.54 dB | 150.8 kb/s (+0.5%), 22.71 dB |
| | 400 kb/s | 363.2 kb/s (−9.2%), 26.49 dB | 398.9 kb/s (−0.3%), 27.41 dB |
| | 1000 kb/s | 1045.3 kb/s (+4.5%), 32.87 dB | 999.7 kb/s (−0.0%), 33.24 dB |
| `vp90-2-12-droppable_1.ivf`, 352x288, 99 frames | 150 kb/s | 155.0 kb/s (+3.3%), 32.06 dB | 150.0 kb/s (+0.0%), 31.84 dB |
| | 400 kb/s | 403.9 kb/s (+1.0%), 36.48 dB | 400.2 kb/s (+0.0%), 37.37 dB |
| | 1000 kb/s | 1011.9 kb/s (+1.2%), 42.20 dB | 1000.0 kb/s (+0.0%), 42.97 dB |
| 226x226 clip forwards and back, 40 frames (`cargo test`) | 300 kb/s | 336.0 kb/s (+12.0%) | 300.0 kb/s (+0.0%) |
| | 1200 kb/s | 1303.9 kb/s (+8.7%) | 1199.6 kb/s (−0.0%) |

One pass overshoots most on short clips: the first key frame is coded
before anything is known about the content, and the overspend is paid back
over the next second.

## How it is checked

- **The WebM project's VP9 test vectors**: bitstreams published with the
  MD5 of every frame the reference decoder shows. `tools/fetch-vectors.sh`
  downloads the 353 `vp90-2-*`, `vp91-2-*`, `vp92-2-*` and `vp93-2-*`
  streams (about 34 MB; the film clips `bbb_`, `sintel_`, `tos_`, 1.8 GB and
  without MD5s, are left out) and `tests/vectors.rs` decodes each and
  compares every shown frame. **All 353 pass**, every frame bit-exact:

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
  | vp91-2-04 | profile 1: 4:2:2, 4:4:0, 4:4:4, and a pre-final 4:4:4 stream | 4 / 4 |
  | vp92-2-20, vp93-2-20 | profiles 2 and 3: 10 / 12-bit, 4:2:0 to 4:4:4 | 8 / 8 |

  `vp91-2-04-yv444.webm` is a stream from before the profile 1 syntax was
  final (its MD5 file is in an older format too): it sizes chroma
  transforms, and the chroma motion vectors of blocks below 8x8, as if its
  4:4:4 chroma were 4:2:0 — see the [specification
  notes](#specification-notes) for how it is recognised.
  **Intra-only frames**: `vp90-2-16-intra-only.webm` passes, and the first
  frame (a key frame) of every vector is bit-exact. Fifteen small vectors
  are committed in [`tests/data`](tests/data/README.md) so `cargo test`
  checks real streams without the download.
- **The encoder** (`tests/encode.rs`), with this crate's decoder as the
  only oracle: every packet decodes, with a fresh decoder, to exactly
  `Encoder::reconstruction()`; at every bit depth (8, 10, 12) and chroma
  format (4:2:0, 4:2:2, 4:4:0, 4:4:4), sizes 1x1, 17x33, 66x34 and
  130x72, searched and fixed partitions, lossless is exact and the lossy
  quantisers keep their PSNR; full-swing 10 / 12-bit content at the finest
  quantisers (coefficients that need the category-6 high bits) decodes
  exactly; PSNR falls and size falls as the quantiser rises at every
  profile; inter frames cost less than intra; rate control meets its
  target (one pass within 15%, two within 5%, on 40 frames); and debug
  builds loop filter the encoder's own reconstruction and compare it with
  the decoded packet sample by sample, every frame.
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

// Encoding: frames in, one packet each out.
let mut cfg = vp9::Config::new(1280, 720);
cfg.bit_depth = 10; // 8, 10 or 12
cfg.chroma = vp9::ChromaFormat::Yuv420; // or Yuv422, Yuv440, Yuv444
cfg.quantizer = 60; // a fixed quantiser, or:
cfg.target_bitrate = Some(2_000_000); // bits per second at cfg.frame_rate
let mut enc = vp9::Encoder::new(cfg.clone());
let packet = enc.encode(&frame)?;
let decoded = enc.reconstruction(); // what a decoder will output

// Two passes: measure the clip first.
let mut first = vp9::encoder::FirstPass::new(cfg.clone());
for f in &frames {
    first.add(f)?;
}
cfg.two_pass = Some(first.finish());
```

`vp9::ivf` reads and writes IVF; `vp9::superframe` splits and builds
superframes. The examples decode IVF to raw planar video with per-frame
MD5s (`cargo run --release --example ivfdec -- in.ivf out.yuv`) and encode
raw planar video to IVF at a quantiser or a bitrate (two-pass), any bit
depth and chroma format (`--example ivfenc -- in.yuv 352 288 out.ivf 60`,
`… out.ivf 500k 10 444`).

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
  blocks not on the block's right edge — surprising, but taken literally it
  matches every vector.
- **Several shown frames in one superframe** (Annex B allows it): the
  vectors expect the last one per packet (`vp90-2-22-svc_1280x720_3.ivf`).
- **Pre-final profile 1 streams** (`vp91-2-04-yv444.webm`): decoded by the
  specification's rules, its key frame leaves 170 000 bits of its tile data
  unread and every frame is wrong. With get_uv_tx_size (and the loop
  filter's chroma transform size) taken from the 4:2:0 chroma block size,
  every frame parses to the end of its data, and with the chroma motion
  vector of a block below 8x8 the average of its four (4:2:0's rule) every
  frame matches its MD5. The decoder decides which rules a non-4:2:0 stream
  follows at each intra frame: the specification's, unless a tile then
  fails to parse or ends with nonzero padding — which 9.2.3 forbids — and
  the older rules parse it with zero padding; inter frames follow the last
  intra frame's choice. Every other vector decodes by the specification's
  rules on the first try.
- **Inferred values are counted** (9.3): a `partition` forced to SPLIT at
  the frame edge, and the implied `mv_hp` / `mv_class0_hp` of 1, are
  counted for adaptation like decoded ones.
