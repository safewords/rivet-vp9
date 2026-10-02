//! Encodes raw 8-bit 4:2:0 planar video (I420: Y, U, V per frame) to IVF.
//!
//! cargo run --release --example ivfenc -- input.yuv WIDTH HEIGHT output.ivf [quantizer]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: ivfenc input.yuv WIDTH HEIGHT output.ivf [quantizer 0-255]");
        std::process::exit(2);
    }
    let raw = std::fs::read(&args[1])?;
    let (w, h): (u32, u32) = (args[2].parse()?, args[3].parse()?);
    let mut cfg = vp9::Config::new(w, h);
    if let Some(q) = args.get(5) {
        cfg.quantizer = q.parse()?;
    }
    let mut enc = vp9::Encoder::new(cfg);
    let mut ivf = vp9::ivf::IvfWriter::new(w as u16, h as u16, 30, 1);
    let template = vp9::Frame::new(w, h, 8, vp9::ChromaFormat::Yuv420);
    let size = template.data.len();
    let mut bytes = 0;
    for (i, chunk) in raw.chunks_exact(size).enumerate() {
        let mut f = template.clone();
        f.data.copy_from_slice(chunk);
        let pkt = enc.encode(&f)?;
        bytes += pkt.len();
        ivf.frame(i as u64, &pkt);
    }
    std::fs::write(&args[4], ivf.finish())?;
    eprintln!("{} frames, {bytes} bytes", raw.len() / size);
    Ok(())
}
