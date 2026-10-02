//! Superframes (Annex B): several frames in one container packet, with an
//! index at the end.

/// Splits a packet into its frames: the frames of a superframe, or the
/// whole packet if it has no (valid) superframe index.
pub fn split(packet: &[u8]) -> Vec<&[u8]> {
    if let Some(sizes) = index(packet) {
        let mut out = Vec::with_capacity(sizes.len());
        let mut pos = 0usize;
        for s in sizes {
            if s == 0 {
                continue;
            }
            out.push(&packet[pos..pos + s]);
            pos += s;
        }
        return out;
    }
    vec![packet]
}

/// The frame sizes of a superframe index, if `packet` ends with one whose
/// sizes fit in the packet.
fn index(packet: &[u8]) -> Option<Vec<usize>> {
    let last = *packet.last()?;
    if last & 0xe0 != 0xc0 {
        return None;
    }
    let sz_bytes = ((last >> 3) & 3) as usize + 1;
    let frames = (last & 7) as usize + 1;
    let index_len = 2 + frames * sz_bytes;
    if packet.len() < index_len || packet[packet.len() - index_len] != last {
        return None;
    }
    let mut p = packet.len() - index_len + 1;
    let mut sizes = Vec::with_capacity(frames);
    let mut total = 0usize;
    for _ in 0..frames {
        // frame_sizes are little-endian.
        let mut s = 0usize;
        for b in 0..sz_bytes {
            s |= (packet[p + b] as usize) << (8 * b);
        }
        p += sz_bytes;
        total += s;
        sizes.push(s);
    }
    if total > packet.len() - index_len {
        return None;
    }
    Some(sizes)
}

/// Builds a superframe from `frames` (the inverse of [`split`]).
pub fn join(frames: &[&[u8]]) -> Vec<u8> {
    assert!(
        !frames.is_empty() && frames.len() <= 8,
        "a superframe holds 1 to 8 frames"
    );
    let max = frames.iter().map(|f| f.len()).max().unwrap();
    let sz_bytes = if max < 1 << 8 {
        1
    } else if max < 1 << 16 {
        2
    } else if max < 1 << 24 {
        3
    } else {
        4
    };
    let marker = 0xc0 | (((sz_bytes - 1) as u8) << 3) | (frames.len() - 1) as u8;
    let mut out = Vec::new();
    for f in frames {
        out.extend_from_slice(f);
    }
    out.push(marker);
    for f in frames {
        for b in 0..sz_bytes {
            out.push((f.len() >> (8 * b)) as u8);
        }
    }
    out.push(marker);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let a = vec![1u8; 300];
        let b = vec![2u8; 5];
        let sf = join(&[&a, &b]);
        let parts = split(&sf);
        assert_eq!(parts, vec![&a[..], &b[..]]);
    }

    #[test]
    fn plain_frame_passes_through() {
        let f = [0x82u8, 0x49, 0x83, 0x42, 0x00];
        assert_eq!(split(&f), vec![&f[..]]);
    }
}
