//! End-to-end throughput: decoding and encoding, frames per second.
//!
//! ```text
//! cargo run --release --example bench -- dec  FILE.{ivf,webm} [--threads N] [--runs R]
//! cargo run --release --example bench -- enc  SRC.yuv W H [--frames N] [--q Q] [--speed S]
//!                                             [--threads N] [--runs R] [--out OUT.ivf]
//! cargo run --release --example bench -- yuv  FILE.{ivf,webm} OUT.yuv [--crop WxH]
//! ```
//!
//! `dec` decodes the whole file `R` times (default 5) and reports the
//! fastest run; `enc` encodes the first `N` frames (default all) of 8-bit
//! 4:2:0 `SRC.yuv` likewise. `yuv` writes the decoded frames as 8-bit
//! planar 4:2:0, cropped to the top-left `W` x `H` — a way to make encoder
//! sources out of the test vectors. `--threads 0` (the default) lets the
//! codec pick; `1` is single-threaded. Set `VP9_FORCE_SCALAR=1` for the
//! scalar kernels.

use std::time::Instant;

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn packets(path: &str) -> Vec<Vec<u8>> {
    let data = std::fs::read(path).expect("read input");
    if data.starts_with(b"DKIF") {
        vp9::ivf::IvfReader::new(&data)
            .unwrap()
            .map(|f| f.unwrap().data.to_vec())
            .collect()
    } else {
        let mut out = Vec::new();
        webm(&data, &mut out, &mut None);
        out
    }
}

fn vint(d: &[u8], pos: usize, keep_marker: bool) -> Option<(u64, usize)> {
    let first = *d.get(pos)?;
    let len = first.leading_zeros() as usize + 1;
    if len > 8 || pos + len > d.len() {
        return None;
    }
    let mut v = if keep_marker {
        first as u64
    } else {
        first as u64 & ((1u64 << (8 - len)) - 1)
    };
    for i in 1..len {
        v = (v << 8) | d[pos + i] as u64;
    }
    Some((v, len))
}

/// Just enough Matroska for the test vectors: the first video track's
/// SimpleBlock / Block payloads.
fn webm(d: &[u8], out: &mut Vec<Vec<u8>>, track: &mut Option<u64>) {
    let mut pos = 0;
    while pos < d.len() {
        let Some((id, il)) = vint(d, pos, true) else {
            return;
        };
        let Some((size, sl)) = vint(d, pos + il, false) else {
            return;
        };
        let start = pos + il + sl;
        let unknown = size == (1u64 << (7 * sl)) - 1;
        if !unknown && start + size as usize > d.len() {
            return;
        }
        let end = if unknown {
            d.len()
        } else {
            start + size as usize
        };
        match id {
            0x18538067 | 0x1F43B675 | 0xA0 => webm(&d[start..end], out, track),
            0xA3 | 0xA1 => {
                let b = &d[start..end];
                if let Some((tn, tl)) = vint(b, 0, false) {
                    if track.is_none() {
                        *track = Some(tn);
                    }
                    if Some(tn) == *track && b.len() >= tl + 3 {
                        out.push(b[tl + 3..].to_vec());
                    }
                }
            }
            _ => {}
        }
        pos = end;
    }
}

fn new_decoder(_threads: usize) -> vp9::Decoder {
    #[allow(unused_mut)]
    let mut dec = vp9::Decoder::new();
    dec.set_threads(_threads); // THREADS
    dec
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("");
    let threads: usize = arg(&args, "--threads", 0);
    let runs: usize = arg(&args, "--runs", 5);
    match mode {
        "dec" => {
            let pkts = packets(&args[2]);
            let mut best = f64::MAX;
            let mut frames = 0;
            let mut size = (0, 0);
            for _ in 0..runs {
                let mut dec = new_decoder(threads);
                let t = Instant::now();
                frames = 0;
                for p in &pkts {
                    for f in dec.decode_all(p).expect("decode") {
                        size = (f.width, f.height);
                        frames += 1;
                    }
                }
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!(
                "decode {}x{} {frames} frames: {:.3} s, {:.2} fps (best of {runs}, threads {threads})",
                size.0,
                size.1,
                best,
                frames as f64 / best
            );
        }
        "enc" => {
            let raw = std::fs::read(&args[2]).expect("read source");
            let (w, h): (u32, u32) = (args[3].parse().unwrap(), args[4].parse().unwrap());
            let tmpl = vp9::Frame::new(w, h, 8, vp9::ChromaFormat::Yuv420);
            let n: usize = arg(&args, "--frames", usize::MAX);
            let frames: Vec<vp9::Frame> = raw
                .chunks_exact(tmpl.data.len())
                .take(n)
                .map(|c| {
                    let mut f = tmpl.clone();
                    f.data.copy_from_slice(c);
                    f
                })
                .collect();
            let mut cfg = vp9::Config::new(w, h);
            cfg.quantizer = arg(&args, "--q", 64);
            cfg.speed = arg(&args, "--speed", 1);
            cfg.threads = threads; // THREADS
            let mut best = f64::MAX;
            let mut bytes = 0;
            let mut out = None;
            for _ in 0..runs {
                let mut enc = vp9::Encoder::new(cfg.clone());
                let mut ivf = vp9::ivf::IvfWriter::new(w as u16, h as u16, 30, 1);
                let t = Instant::now();
                bytes = 0;
                for (i, f) in frames.iter().enumerate() {
                    let p = enc.encode(f).expect("encode");
                    bytes += p.len();
                    ivf.frame(i as u64, &p);
                }
                best = best.min(t.elapsed().as_secs_f64());
                out = Some(ivf.finish());
            }
            if let (Some(path), Some(o)) = (
                args.iter()
                    .position(|a| a == "--out")
                    .and_then(|i| args.get(i + 1)),
                out,
            ) {
                std::fs::write(path, o).expect("write output");
            }
            println!(
                "encode {w}x{h} {} frames q{} speed {}: {:.3} s, {:.3} fps, {bytes} bytes (best of {runs}, threads {threads})",
                frames.len(),
                cfg.quantizer,
                cfg.speed,
                best,
                frames.len() as f64 / best
            );
        }
        "yuv" => {
            let pkts = packets(&args[2]);
            let crop: String = arg(&args, "--crop", String::new());
            let mut out = Vec::new();
            let mut dec = vp9::Decoder::new();
            let mut n = 0;
            for p in &pkts {
                for f in dec.decode_all(p).expect("decode") {
                    let (cw, ch) = crop
                        .split_once('x')
                        .map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()))
                        .unwrap_or((f.width, f.height));
                    for pl in 0..3 {
                        let (pw, ph) = if pl == 0 { (cw, ch) } else { (cw / 2, ch / 2) };
                        for y in 0..ph {
                            for x in 0..pw {
                                out.push(f.sample(pl, x, y) as u8);
                            }
                        }
                    }
                    n += 1;
                }
            }
            std::fs::write(&args[3], out).expect("write yuv");
            eprintln!("{n} frames");
        }
        _ => {
            eprintln!(
                "usage: bench dec FILE | enc SRC.yuv W H | yuv FILE OUT.yuv (see the source)"
            );
            std::process::exit(2);
        }
    }
}
