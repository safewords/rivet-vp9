//! The encoder: every packet must decode, lossless must be exact, and lossy
//! quality must track the quantiser. The natural-content source is the
//! decoded frames of a committed test vector (tests/data).

mod common;

use vp9::{ChromaFormat, Config, Decoder, Encoder, Frame};

/// PSNR of plane `plane`, against the peak of the frames' bit depth.
fn psnr(a: &Frame, b: &Frame, plane: usize) -> f64 {
    let pl = a.planes[plane];
    let mut se = 0.0;
    for y in 0..pl.height {
        for x in 0..pl.width {
            se += (a.sample(plane, x, y) as f64 - b.sample(plane, x, y) as f64).powi(2);
        }
    }
    let mse = se / (pl.width * pl.height) as f64;
    let peak = ((1u32 << a.bit_depth) - 1) as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mse).log10()
    }
}

const FORMATS: [ChromaFormat; 4] = [
    ChromaFormat::Yuv420,
    ChromaFormat::Yuv422,
    ChromaFormat::Yuv440,
    ChromaFormat::Yuv444,
];

/// `f` (8-bit 4:2:0) at another bit depth and chroma format: chroma
/// resampled by nearest neighbour, samples widened with the 2x2 mean so
/// the extra low bits carry detail.
fn convert(f: &Frame, bit_depth: u32, chroma: ChromaFormat) -> Frame {
    let mut g = Frame::new(f.width, f.height, bit_depth, chroma);
    let (sx, sy) = chroma.shifts();
    for p in 0..3 {
        let src = f.planes[p];
        let dst = g.planes[p];
        for y in 0..dst.height {
            for x in 0..dst.width {
                // Position in the source plane.
                let (fx, fy) = if p == 0 {
                    (x, y)
                } else {
                    ((x << sx) >> 1, (y << sy) >> 1)
                };
                let at = |dx: u32, dy: u32| {
                    f.sample(
                        p,
                        (fx + dx).min(src.width - 1),
                        (fy + dy).min(src.height - 1),
                    ) as u32
                };
                let sum = at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1);
                let v = (sum << (bit_depth - 8)) >> 2;
                g.set_sample(p, x, y, v.min((1 << bit_depth) - 1) as u16);
            }
        }
    }
    g
}

/// The profile in a frame's first byte (frame_marker, profile_low_bit,
/// profile_high_bit).
fn profile_of(packet: &[u8]) -> u8 {
    ((packet[0] >> 5) & 1) | (((packet[0] >> 4) & 1) << 1)
}

/// The 10 frames of vp90-2-03-size-226x226.webm, decoded: natural video.
fn natural() -> Vec<Frame> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/vp90-2-03-size-226x226.webm");
    let mut d = Decoder::new();
    common::packets(&p)
        .iter()
        .filter_map(|pk| d.decode(pk).unwrap())
        .collect()
}

fn synthetic(w: u32, h: u32, t: u32) -> Frame {
    let mut f = Frame::new(w, h, 8, ChromaFormat::Yuv420);
    for p in 0..3 {
        let pl = f.planes[p];
        for y in 0..pl.height {
            for x in 0..pl.width {
                // Luma pans 2 samples a frame, chroma (half resolution) 1.
                let xs = if p == 0 { x + 2 * t } else { x + t };
                let v = if p == 0 {
                    let ring = ((xs as f64 - 20.0).hypot(y as f64 - 14.0) / 3.0).sin() * 50.0;
                    (110.0 + ring + ((xs * 7 + y * 3) % 23) as f64) as u16
                } else {
                    (90 + (xs * 2 + y + p as u32 * 17) % 60) as u16
                };
                f.set_sample(p, x, y, v.min(255));
            }
        }
    }
    f
}

/// Encodes `frames`, decodes the packets with a fresh decoder, returns the
/// decoded frames and the packet sizes.
fn round_trip(cfg: Config, frames: &[Frame]) -> (Vec<Frame>, Vec<usize>) {
    let profile = cfg.profile();
    let mut enc = Encoder::new(cfg);
    let mut dec = Decoder::new();
    let mut out = Vec::new();
    let mut sizes = Vec::new();
    for f in frames {
        let pkt = enc.encode(f).unwrap();
        assert_eq!(profile_of(&pkt), profile);
        sizes.push(pkt.len());
        let d = dec
            .decode(&pkt)
            .unwrap()
            .expect("every packet shows a frame");
        // The decoded frame is the encoder's reconstruction, byte for byte.
        assert!(
            enc.reconstruction() == Some(&d),
            "decoded frame differs from the encoder's reconstruction"
        );
        out.push(d);
    }
    (out, sizes)
}

#[test]
fn lossless_natural_is_exact() {
    let src = natural();
    let mut cfg = Config::new(src[0].width, src[0].height);
    cfg.quantizer = 0;
    cfg.keyframe_interval = 5;
    let (out, sizes) = round_trip(cfg, &src);
    for (a, b) in src.iter().zip(&out) {
        assert_eq!(a.data, b.data);
    }
    eprintln!(
        "lossless 226x226: {} bytes for {} frames",
        sizes.iter().sum::<usize>(),
        sizes.len()
    );
}

#[test]
fn quality_tracks_the_quantizer() {
    let src = natural();
    let mut last_psnr = f64::INFINITY;
    let mut last_bytes = usize::MAX;
    for (q, min_psnr) in [
        (16u8, 44.0),
        (48, 38.0),
        (96, 33.0),
        (160, 29.0),
        (240, 21.0),
    ] {
        let mut cfg = Config::new(src[0].width, src[0].height);
        cfg.quantizer = q;
        let (out, sizes) = round_trip(cfg, &src);
        let y: f64 = src
            .iter()
            .zip(&out)
            .map(|(a, b)| psnr(a, b, 0))
            .sum::<f64>()
            / src.len() as f64;
        let u: f64 = src
            .iter()
            .zip(&out)
            .map(|(a, b)| psnr(a, b, 1))
            .sum::<f64>()
            / src.len() as f64;
        let v: f64 = src
            .iter()
            .zip(&out)
            .map(|(a, b)| psnr(a, b, 2))
            .sum::<f64>()
            / src.len() as f64;
        let bytes: usize = sizes.iter().sum();
        eprintln!(
            "q {q:3}: {bytes:6} bytes ({} key, {:.0} per inter frame), PSNR Y {y:.2} U {u:.2} V {v:.2} dB",
            sizes[0],
            sizes[1..].iter().sum::<usize>() as f64 / (sizes.len() - 1) as f64
        );
        assert!(y > min_psnr, "q {q}: Y PSNR {y:.2} below {min_psnr}");
        assert!(
            y < last_psnr && bytes < last_bytes,
            "q {q}: quality or size did not fall"
        );
        last_psnr = y;
        last_bytes = bytes;
    }
}

#[test]
fn inter_frames_are_cheaper_than_key_frames() {
    let src = natural();
    let mut cfg = Config::new(src[0].width, src[0].height);
    cfg.quantizer = 64;
    let (_, inter) = round_trip(cfg.clone(), &src);
    cfg.keyframe_interval = 1;
    let (_, intra) = round_trip(cfg, &src);
    let a: usize = inter.iter().sum();
    let b: usize = intra.iter().sum();
    eprintln!("10 frames at q 64: {a} bytes with inter frames, {b} all-intra");
    assert!(a * 10 < b * 8, "inter coding saved less than 20%");
}

#[test]
fn every_size_and_block_size_decodes() {
    for (w, h) in [
        (1, 1),
        (2, 3),
        (8, 8),
        (9, 7),
        (17, 33),
        (64, 64),
        (65, 1),
        (130, 66),
        (200, 72),
    ] {
        for bs in [8, 16, 32, 64] {
            let frames: Vec<Frame> = (0..3).map(|t| synthetic(w, h, t)).collect();
            let mut cfg = Config::new(w, h);
            cfg.block_size = bs;
            cfg.quantizer = 40;
            let (out, _) = round_trip(cfg.clone(), &frames);
            let y = psnr(&frames[2], &out[2], 0);
            assert!(y > 30.0, "{w}x{h} blocks {bs}: PSNR {y:.1}");
            cfg.quantizer = 0;
            let (out, _) = round_trip(cfg, &frames);
            for (a, b) in frames.iter().zip(&out) {
                assert_eq!(a.data, b.data, "{w}x{h} blocks {bs}: lossless mismatch");
            }
        }
    }
}

#[test]
fn moving_content_uses_motion() {
    // A panning picture: inter frames must be far smaller than the key frame.
    let frames: Vec<Frame> = (0..6).map(|t| synthetic(96, 64, t)).collect();
    let mut cfg = Config::new(96, 64);
    cfg.quantizer = 48;
    let (out, sizes) = round_trip(cfg, &frames);
    eprintln!("panning 96x64: sizes {sizes:?}");
    assert!(sizes[1..].iter().all(|&s| s * 3 < sizes[0]), "{sizes:?}");
    for (a, b) in frames.iter().zip(&out) {
        assert!(psnr(a, b, 0) > 34.0);
    }
}

#[test]
fn bad_input_is_refused() {
    let f = Frame::new(16, 16, 8, ChromaFormat::Yuv420);
    assert!(Config::new(0, 10).validate().is_err());
    assert!(Encoder::new(Config::new(0, 10)).encode(&f).is_err());
    let mut cfg = Config::new(16, 16);
    cfg.block_size = 12;
    assert!(Encoder::new(cfg).encode(&f).is_err());
    let mut enc = Encoder::new(Config::new(16, 16));
    assert!(enc.encode(&f).is_ok());
    assert!(
        enc.encode(&Frame::new(32, 16, 8, ChromaFormat::Yuv420))
            .is_err()
    );
    assert!(
        enc.encode(&Frame::new(16, 16, 10, ChromaFormat::Yuv420))
            .is_err()
    );
    assert!(
        enc.encode(&Frame::new(16, 16, 8, ChromaFormat::Yuv444))
            .is_err()
    );
    let mut cfg = Config::new(16, 16);
    cfg.bit_depth = 9;
    assert!(cfg.validate().is_err());
    let mut cfg = Config::new(16, 16);
    cfg.color_space = vp9::ColorSpace::Rgb;
    assert!(cfg.validate().is_err());
    cfg.chroma = ChromaFormat::Yuv444;
    assert!(cfg.validate().is_ok());
}

fn synthetic_fmt(w: u32, h: u32, t: u32, bit_depth: u32, chroma: ChromaFormat) -> Frame {
    convert(&synthetic(w, h, t), bit_depth, chroma)
}

#[test]
fn every_profile_round_trips() {
    // Profiles 0-3: every bit depth and chroma format, odd sizes (forced
    // edge partitions, which 4:2:2 and 4:4:0 must split where HORZ / VERT
    // would give a forbidden chroma block), lossless and lossy.
    for bd in [8, 10, 12] {
        for chroma in FORMATS {
            for (w, h) in [(1, 1), (17, 33), (66, 34), (130, 72)] {
                for bs in [16, 64] {
                    let frames: Vec<Frame> =
                        (0..3).map(|t| synthetic_fmt(w, h, t, bd, chroma)).collect();
                    let mut cfg = Config::new(w, h);
                    cfg.bit_depth = bd;
                    cfg.chroma = chroma;
                    cfg.block_size = bs;
                    let what = format!("{bd}-bit {chroma:?} {w}x{h} blocks {bs}");
                    cfg.quantizer = 0;
                    let (out, _) = round_trip(cfg.clone(), &frames);
                    for (a, b) in frames.iter().zip(&out) {
                        assert_eq!(a.data, b.data, "{what}: lossless mismatch");
                    }
                    for q in [40u8, 160] {
                        cfg.quantizer = q;
                        let (out, _) = round_trip(cfg.clone(), &frames);
                        let y = psnr(&frames[2], &out[2], 0);
                        let min = if q == 40 { 30.0 } else { 20.0 };
                        assert!(y > min, "{what} q {q}: PSNR {y:.1}");
                    }
                }
            }
        }
    }
}

#[test]
fn quality_at_every_profile() {
    // The natural clip in each format: PSNR per plane at several
    // quantisers, falling with the quantiser.
    let src = natural();
    for bd in [8, 10, 12] {
        for chroma in FORMATS {
            let frames: Vec<Frame> = src[..4].iter().map(|f| convert(f, bd, chroma)).collect();
            let mut last = f64::INFINITY;
            for q in [32u8, 96, 192] {
                let mut cfg = Config::new(frames[0].width, frames[0].height);
                cfg.bit_depth = bd;
                cfg.chroma = chroma;
                cfg.quantizer = q;
                let (out, sizes) = round_trip(cfg, &frames);
                let mean = |p: usize| {
                    frames
                        .iter()
                        .zip(&out)
                        .map(|(a, b)| psnr(a, b, p))
                        .sum::<f64>()
                        / frames.len() as f64
                };
                let (y, u, v) = (mean(0), mean(1), mean(2));
                eprintln!(
                    "{bd:2}-bit {chroma:?} q {q:3}: {:6} bytes, PSNR Y {y:.2} U {u:.2} V {v:.2} dB",
                    sizes.iter().sum::<usize>()
                );
                assert!(y < last && y > 24.0, "{bd}-bit {chroma:?} q {q}: {y:.2}");
                last = y;
            }
        }
    }
}

#[test]
fn encoder_reconstruction_is_the_decoders() {
    // Without the loop filter, debug builds of the encoder compare their
    // own reconstruction with the decoded packet, sample by sample.
    let src = natural();
    for q in [1u8, 80, 200] {
        let mut cfg = Config::new(src[0].width, src[0].height);
        cfg.quantizer = q;
        cfg.loop_filter_level = Some(0);
        cfg.block_size = if q == 80 { 32 } else { 16 };
        round_trip(cfg, &src[..4]);
    }
}

#[test]
fn high_bit_depth_large_coefficients() {
    // Full-swing checkerboards at the finest quantisers: coefficients past
    // the 8-bit range of a category-6 token (16450), which need its extra
    // high bits at 10 and 12 bits.
    for bd in [10u32, 12] {
        let mut f = Frame::new(128, 64, bd, ChromaFormat::Yuv444);
        for p in 0..3 {
            for y in 0..64 {
                for x in 0..128 {
                    let on = ((x / 32) + (y / 32)) % 2 == 0;
                    f.set_sample(p, x, y, if on { (1u16 << bd) - 1 } else { 0 });
                }
            }
        }
        for q in [1u8, 4] {
            let mut cfg = Config::new(128, 64);
            cfg.bit_depth = bd;
            cfg.chroma = ChromaFormat::Yuv444;
            cfg.quantizer = q;
            cfg.block_size = 32;
            let (out, _) = round_trip(cfg, &[f.clone(), f.clone()]);
            for o in &out {
                assert!(psnr(&f, o, 0) > 60.0, "{bd}-bit q {q}");
            }
        }
    }
}

/// Encodes `frames` at `bitrate` (30 frames/s), one or two passes; returns
/// the achieved bitrate and the mean luma PSNR.
fn rate_controlled(frames: &[Frame], bitrate: u64, two_pass: bool) -> (f64, f64, Vec<usize>) {
    let mut cfg = Config::new(frames[0].width, frames[0].height);
    cfg.bit_depth = frames[0].bit_depth;
    cfg.chroma = frames[0].chroma;
    cfg.target_bitrate = Some(bitrate);
    cfg.frame_rate = 30.0;
    if two_pass {
        let mut fp = vp9::encoder::FirstPass::new(cfg.clone());
        for f in frames {
            fp.add(f).unwrap();
        }
        cfg.two_pass = Some(fp.finish());
    }
    let (out, sizes) = round_trip(cfg, frames);
    let bits = sizes.iter().sum::<usize>() as f64 * 8.0;
    let rate = bits * 30.0 / frames.len() as f64;
    let y = frames
        .iter()
        .zip(&out)
        .map(|(a, b)| psnr(a, b, 0))
        .sum::<f64>()
        / frames.len() as f64;
    (rate, y, sizes)
}

/// The natural clip played forwards then backwards, `n` frames.
fn ping_pong(src: &[Frame], n: usize) -> Vec<Frame> {
    let period = 2 * src.len() - 2;
    (0..n)
        .map(|i| {
            let k = i % period;
            src[if k < src.len() { k } else { period - k }].clone()
        })
        .collect()
}

#[test]
fn rate_control_meets_the_target() {
    // 40 frames of natural video (the clip forwards and back), 226x226.
    let frames = ping_pong(&natural(), 40);
    for kbps in [300u64, 1200] {
        for two in [false, true] {
            let (rate, y, sizes) = rate_controlled(&frames, kbps * 1000, two);
            let err = rate / (kbps * 1000) as f64 - 1.0;
            eprintln!(
                "{kbps} kb/s {}: {:.1} kb/s ({:+.1}%), PSNR Y {y:.2}, key frame {} bytes",
                if two { "two-pass" } else { "one-pass" },
                rate / 1000.0,
                err * 100.0,
                sizes[0]
            );
            let tol = if two { 0.05 } else { 0.15 };
            assert!(err.abs() < tol, "{kbps} kb/s two-pass {two}: {rate:.0}");
        }
    }
    // More bits, better pictures.
    let (_, lo, _) = rate_controlled(&frames[..12], 300_000, false);
    let (_, hi, _) = rate_controlled(&frames[..12], 1_200_000, false);
    assert!(hi > lo + 3.0, "{lo:.2} -> {hi:.2}");
}

#[test]
fn rate_control_at_high_bit_depth() {
    let frames: Vec<Frame> = ping_pong(&natural(), 16)
        .iter()
        .map(|f| convert(f, 10, ChromaFormat::Yuv444))
        .collect();
    let (rate, y, _) = rate_controlled(&frames, 800_000, true);
    eprintln!("10-bit 4:4:4 at 800 kb/s two-pass: {rate:.0} b/s, PSNR Y {y:.2}");
    assert!((rate / 800_000.0 - 1.0).abs() < 0.05);
}

#[test]
#[ignore = "measurement on downloaded vectors (tools/fetch-vectors.sh)"]
fn rate_control_long_clips() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors");
    for (name, n) in [
        ("vp90-2-09-aq2.webm", 100),
        ("vp90-2-12-droppable_1.ivf", 99),
    ] {
        let p = dir.join(name);
        if !p.exists() {
            eprintln!("{name} not downloaded; skipped");
            continue;
        }
        let mut d = Decoder::new();
        let frames: Vec<Frame> = common::packets(&p)
            .iter()
            .filter_map(|pk| d.decode(pk).unwrap())
            .take(n)
            .collect();
        for kbps in [150u64, 400, 1000] {
            for two in [false, true] {
                let (rate, y, _) = rate_controlled(&frames, kbps * 1000, two);
                eprintln!(
                    "{name} {}x{} {} frames, {kbps} kb/s {}: {:.1} kb/s ({:+.1}%), PSNR Y {y:.2}",
                    frames[0].width,
                    frames[0].height,
                    frames.len(),
                    if two { "two-pass" } else { "one-pass" },
                    rate / 1000.0,
                    (rate / (kbps * 1000) as f64 - 1.0) * 100.0
                );
            }
        }
    }
}

/// Bjøntegaard delta rate: the average bitrate difference (%) of `b`
/// against `a` at equal PSNR, each a list of (bytes, PSNR) points;
/// piecewise-linear log-rate over the PSNR range both cover.
fn bd_rate(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let curve = |v: &[(f64, f64)]| {
        let mut c: Vec<(f64, f64)> = v.iter().map(|&(r, p)| (p, r.ln())).collect();
        c.sort_by(|x, y| x.0.total_cmp(&y.0));
        c
    };
    let (ca, cb) = (curve(a), curve(b));
    let lo = ca[0].0.max(cb[0].0);
    let hi = ca[ca.len() - 1].0.min(cb[cb.len() - 1].0);
    let at = |c: &[(f64, f64)], p: f64| {
        let i = c.windows(2).position(|w| p <= w[1].0).unwrap_or(c.len() - 2);
        let (p0, r0) = c[i];
        let (p1, r1) = c[i + 1];
        r0 + (r1 - r0) * (p - p0) / (p1 - p0)
    };
    let n = 200;
    let mut d = 0.0;
    for k in 0..=n {
        let p = lo + (hi - lo) * k as f64 / n as f64;
        d += at(&cb, p) - at(&ca, p);
    }
    ((d / (n + 1) as f64).exp() - 1.0) * 100.0
}

#[test]
#[ignore = "measurement: size at equal quality and time per speed"]
fn tool_gains() {
    let mut clips = vec![("226x226 natural, 10 frames", natural())];
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors/vp90-2-12-droppable_1.ivf");
    if p.exists() {
        let mut d = Decoder::new();
        let f: Vec<Frame> = common::packets(&p)
            .iter()
            .filter_map(|pk| d.decode(pk).unwrap())
            .take(20)
            .collect();
        clips.push(("352x288 droppable_1, 20 frames", f));
    }
    for (name, frames) in &clips {
        let mut base: Option<Vec<(f64, f64)>> = None;
        for (label, speed, bs) in [
            ("speed 2, 16x16", 2u8, 16u32),
            ("speed 2, 32x32", 2, 32),
            ("speed 1", 1, 16),
            ("speed 0", 0, 16),
        ] {
            let t = std::time::Instant::now();
            let pts: Vec<(f64, f64)> = [40u8, 80, 120, 160, 200]
                .iter()
                .map(|&q| {
                    let mut cfg = Config::new(frames[0].width, frames[0].height);
                    cfg.quantizer = q;
                    cfg.speed = speed;
                    cfg.block_size = bs;
                    let (out, sizes) = round_trip(cfg, frames);
                    let y = frames
                        .iter()
                        .zip(&out)
                        .map(|(a, b)| psnr(a, b, 0))
                        .sum::<f64>()
                        / frames.len() as f64;
                    (sizes.iter().sum::<usize>() as f64, y)
                })
                .collect();
            let fps = 5.0 * frames.len() as f64 / t.elapsed().as_secs_f64();
            let bd = base.as_ref().map_or(0.0, |b| bd_rate(b, &pts));
            eprintln!("{name}: {label}: BD-rate {bd:+.1}% vs speed 2 16x16, {fps:.1} frames/s, points {pts:.0?}");
            if base.is_none() {
                base = Some(pts);
            }
        }
    }
}
