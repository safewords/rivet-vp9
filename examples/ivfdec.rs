//! Decodes an IVF file to raw planar video (Y, U, V per frame; 16-bit
//! little-endian samples above 8 bits) and prints each frame's MD5.
//!
//! cargo run --release --example ivfdec -- input.ivf [output.yuv]

use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: ivfdec input.ivf [output.yuv]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1])?;
    let mut out = match args.get(2) {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => None,
    };
    let mut dec = vp9::Decoder::new();
    let mut n = 0;
    for pkt in vp9::ivf::IvfReader::new(&data)? {
        if let Some(f) = dec.decode(pkt?.data)? {
            println!(
                "{:x}  frame {n:05} {}x{} {}-bit {:?}",
                md5_hex(f.packed()),
                f.width,
                f.height,
                f.bit_depth,
                f.chroma
            );
            if let Some(o) = out.as_mut() {
                o.write_all(f.packed())?;
            }
            n += 1;
        }
    }
    eprintln!("{n} frames");
    Ok(())
}

/// A tiny MD5 so the example needs nothing but this crate (RFC 1321).
fn md5_hex(data: &[u8]) -> Md5 {
    let s: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64)
        .map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32)
        .collect();
    let mut h = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_le_bytes());
    for chunk in msg.chunks(64) {
        let m: Vec<u32> = chunk
            .chunks(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let (mut a, mut b, mut c, mut d) = (h[0], h[1], h[2], h[3]);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let t = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(k[i])
                    .wrapping_add(m[g])
                    .rotate_left(s[i]),
            );
            a = t;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
    }
    Md5(h)
}

struct Md5([u32; 4]);

impl std::fmt::LowerHex for Md5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for w in self.0 {
            for b in w.to_le_bytes() {
                write!(f, "{b:02x}")?;
            }
        }
        Ok(())
    }
}
