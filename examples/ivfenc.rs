//! Encodes raw planar video (Y, U, V per frame; 8-bit samples one byte
//! each, 10 / 12-bit little-endian u16) to IVF.
//!
//! cargo run --release --example ivfenc -- input.yuv WIDTH HEIGHT output.ivf [RATE] [BITS] [CHROMA]
//!
//! RATE is a quantiser 0-255 (default 64) or a bitrate such as `500k`
//! (two-pass); BITS is 8, 10 or 12 (default 8); CHROMA is 420, 422, 440 or
//! 444 (default 420).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: ivfenc input.yuv WIDTH HEIGHT output.ivf [quantizer 0-255 | bitrate e.g. 500k] [8|10|12] [420|422|440|444]"
        );
        std::process::exit(2);
    }
    let raw = std::fs::read(&args[1])?;
    let (w, h): (u32, u32) = (args[2].parse()?, args[3].parse()?);
    let bits: u32 = args.get(6).map_or(Ok(8), |s| s.parse())?;
    let chroma = match args.get(7).map_or("420", |s| s.as_str()) {
        "420" => vp9::ChromaFormat::Yuv420,
        "422" => vp9::ChromaFormat::Yuv422,
        "440" => vp9::ChromaFormat::Yuv440,
        "444" => vp9::ChromaFormat::Yuv444,
        other => return Err(format!("unknown chroma format {other}").into()),
    };
    let mut cfg = vp9::Config::new(w, h);
    cfg.bit_depth = bits;
    cfg.chroma = chroma;
    let template = vp9::Frame::new(w, h, bits, chroma);
    let size = template.data.len();
    let frames: Vec<vp9::Frame> = raw
        .chunks_exact(size)
        .map(|chunk| {
            let mut f = template.clone();
            f.data.copy_from_slice(chunk);
            f
        })
        .collect();
    match args.get(5) {
        Some(r) if r.ends_with('k') => {
            cfg.target_bitrate = Some(r.trim_end_matches('k').parse::<u64>()? * 1000);
            let mut first = vp9::encoder::FirstPass::new(cfg.clone());
            for f in &frames {
                first.add(f)?;
            }
            cfg.two_pass = Some(first.finish());
        }
        Some(q) => cfg.quantizer = q.parse()?,
        None => {}
    }
    let mut enc = vp9::Encoder::new(cfg);
    let mut ivf = vp9::ivf::IvfWriter::new(w as u16, h as u16, 30, 1);
    let mut bytes = 0;
    for (i, f) in frames.iter().enumerate() {
        let pkt = enc.encode(f)?;
        bytes += pkt.len();
        ivf.frame(i as u64, &pkt);
    }
    std::fs::write(&args[4], ivf.finish())?;
    eprintln!(
        "{} frames, {bytes} bytes ({:.1} kb/s at 30 frames/s)",
        frames.len(),
        bytes as f64 * 8.0 * 30.0 / frames.len().max(1) as f64 / 1000.0
    );
    Ok(())
}
