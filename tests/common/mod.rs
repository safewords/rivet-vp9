//! Test support: a minimal WebM (Matroska) block extractor and the test
//! vector runner. Shared by the integration tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// The frames of the first video track of a WebM file, in file order.
///
/// Just enough EBML for the test vectors: descends Segment, Cluster and
/// BlockGroup, takes the payload of every SimpleBlock and Block (no
/// lacing), and skips everything else.
pub fn webm_frames(data: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    walk(data, &mut out, &mut None);
    out
}

fn vint(data: &[u8], pos: usize, keep_marker: bool) -> Option<(u64, usize)> {
    let first = *data.get(pos)?;
    let len = first.leading_zeros() as usize + 1;
    if len > 8 || pos + len > data.len() {
        return None;
    }
    let mut v = if keep_marker { first as u64 } else { (first as u64) & ((1u64 << (8 - len)) - 1) };
    for i in 1..len {
        v = (v << 8) | data[pos + i] as u64;
    }
    Some((v, len))
}

fn walk(data: &[u8], out: &mut Vec<Vec<u8>>, track: &mut Option<u64>) {
    let mut pos = 0;
    while pos < data.len() {
        let Some((id, il)) = vint(data, pos, true) else { return };
        let Some((size, sl)) = vint(data, pos + il, false) else { return };
        let start = pos + il + sl;
        let unknown = size == (1u64 << (7 * sl)) - 1;
        if !unknown && start + size as usize > data.len() {
            // An element larger than its parent: the file is corrupt from
            // here (vp90-2-15-fuzz-flicker.webm ends with one). Stop.
            return;
        }
        let end = if unknown { data.len() } else { start + size as usize };
        match id {
            // Segment, Cluster, BlockGroup: descend.
            0x18538067 | 0x1F43B675 | 0xA0 => walk(&data[start..end], out, track),
            // SimpleBlock, Block.
            0xA3 | 0xA1 => {
                let b = &data[start..end];
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

/// The packets of a test vector (WebM or IVF).
pub fn packets(path: &Path) -> Vec<Vec<u8>> {
    let data = std::fs::read(path).unwrap();
    // By content, not extension: vp90-2-13-largescaling.ivf is WebM.
    if data.starts_with(b"DKIF") {
        vp9::ivf::IvfReader::new(&data).unwrap().map(|f| f.unwrap().data.to_vec()).collect()
    } else {
        webm_frames(&data)
    }
}

/// The expected MD5s of a vector's shown frames.
pub fn expected_md5s(path: &Path) -> Vec<String> {
    let md5 = PathBuf::from(format!("{}.md5", path.display()));
    std::fs::read_to_string(md5)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().next().unwrap().to_string())
        .collect()
}

/// Outcome of decoding one vector.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub name: String,
    /// Frames decoded and shown.
    pub frames: usize,
    /// Expected shown frames.
    pub expected: usize,
    /// Leading frames whose MD5 matched.
    pub matched: usize,
    /// First frame that did not match (or the error).
    pub failure: Option<String>,
}

impl Outcome {
    pub fn passed(&self) -> bool {
        self.failure.is_none() && self.matched == self.expected && self.frames == self.expected
    }
}

/// Decodes a vector and compares every shown frame with its MD5.
pub fn run_vector(path: &Path) -> Outcome {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let expected = expected_md5s(path);
    let mut dec = vp9::Decoder::new();
    let mut frames = 0usize;
    let mut matched = 0usize;
    let mut failure = None;
    for (i, p) in packets(path).iter().enumerate() {
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dec.decode(p)));
        let out = match out {
            Ok(Ok(v)) => v.into_iter().collect::<Vec<_>>(),
            Ok(Err(e)) => {
                failure = Some(format!("packet {i}: {e}"));
                break;
            }
            Err(_) => {
                failure = Some(format!("packet {i}: panic"));
                break;
            }
        };
        for f in out {
            let sum = format!("{:x}", md5::compute(f.packed()));
            if failure.is_none() {
                if expected.get(frames) == Some(&sum) {
                    matched += 1;
                } else {
                    failure = Some(format!("frame {frames} ({}x{}) md5 mismatch", f.width, f.height));
                }
            }
            frames += 1;
        }
        if failure.is_some() {
            break;
        }
    }
    if failure.is_none() && frames != expected.len() {
        failure = Some(format!("{} frames shown, {} expected", frames, expected.len()));
    }
    Outcome { name, frames, expected: expected.len(), matched, failure }
}

/// The downloaded vectors (see tools/fetch-vectors.sh), sorted.
pub fn downloaded_vectors() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors");
    let Ok(rd) = std::fs::read_dir(&dir) else { return Vec::new() };
    let mut v: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "webm" || e == "ivf"))
        .filter(|p| PathBuf::from(format!("{}.md5", p.display())).exists())
        .collect();
    v.sort();
    v
}

/// Runs vectors on all cores.
pub fn run_all(paths: &[PathBuf]) -> Vec<Outcome> {
    let n = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..n {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= paths.len() {
                        break;
                    }
                    let o = run_vector(&paths[i]);
                    results.lock().unwrap().push(o);
                }
            });
        }
    });
    let mut r = results.into_inner().unwrap();
    r.sort_by(|a, b| a.name.cmp(&b.name));
    r
}
