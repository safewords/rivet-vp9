//! The encoder: every packet must decode, lossless must be exact, and lossy
//! quality must track the quantiser. The natural-content source is the
//! decoded frames of a committed test vector (tests/data).

mod common;

use vp9::{ChromaFormat, Config, Decoder, Encoder, Frame};

fn psnr(a: &Frame, b: &Frame, plane: usize) -> f64 {
    let (x, y) = (a.plane(plane), b.plane(plane));
    let se: f64 = x.iter().zip(y).map(|(&p, &q)| (p as f64 - q as f64).powi(2)).sum();
    let mse = se / x.len() as f64;
    if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() }
}

/// The 10 frames of vp90-2-03-size-226x226.webm, decoded: natural video.
fn natural() -> Vec<Frame> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/vp90-2-03-size-226x226.webm");
    let mut d = Decoder::new();
    common::packets(&p).iter().filter_map(|pk| d.decode(pk).unwrap()).collect()
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
    let mut enc = Encoder::new(cfg);
    let mut dec = Decoder::new();
    let mut out = Vec::new();
    let mut sizes = Vec::new();
    for f in frames {
        let pkt = enc.encode(f).unwrap();
        sizes.push(pkt.len());
        out.push(dec.decode(&pkt).unwrap().expect("every packet shows a frame"));
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
    eprintln!("lossless 226x226: {} bytes for {} frames", sizes.iter().sum::<usize>(), sizes.len());
}

#[test]
fn quality_tracks_the_quantizer() {
    let src = natural();
    let mut last_psnr = f64::INFINITY;
    let mut last_bytes = usize::MAX;
    for (q, min_psnr) in [(16u8, 44.0), (48, 38.0), (96, 33.0), (160, 29.0), (240, 21.0)] {
        let mut cfg = Config::new(src[0].width, src[0].height);
        cfg.quantizer = q;
        let (out, sizes) = round_trip(cfg, &src);
        let y: f64 = src.iter().zip(&out).map(|(a, b)| psnr(a, b, 0)).sum::<f64>() / src.len() as f64;
        let u: f64 = src.iter().zip(&out).map(|(a, b)| psnr(a, b, 1)).sum::<f64>() / src.len() as f64;
        let v: f64 = src.iter().zip(&out).map(|(a, b)| psnr(a, b, 2)).sum::<f64>() / src.len() as f64;
        let bytes: usize = sizes.iter().sum();
        eprintln!(
            "q {q:3}: {bytes:6} bytes ({} key, {:.0} per inter frame), PSNR Y {y:.2} U {u:.2} V {v:.2} dB",
            sizes[0],
            sizes[1..].iter().sum::<usize>() as f64 / (sizes.len() - 1) as f64
        );
        assert!(y > min_psnr, "q {q}: Y PSNR {y:.2} below {min_psnr}");
        assert!(y < last_psnr && bytes < last_bytes, "q {q}: quality or size did not fall");
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
    for (w, h) in [(1, 1), (2, 3), (8, 8), (9, 7), (17, 33), (64, 64), (65, 1), (130, 66), (200, 72)] {
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
    assert!(enc.encode(&Frame::new(32, 16, 8, ChromaFormat::Yuv420)).is_err());
    assert!(enc.encode(&Frame::new(16, 16, 10, ChromaFormat::Yuv420)).is_err());
    assert!(enc.encode(&Frame::new(16, 16, 8, ChromaFormat::Yuv444)).is_err());
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
