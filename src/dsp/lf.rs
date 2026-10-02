//! Sample filtering of the loop filter (8.8.5): the masks, the narrow
//! filter and the wide filters, for one position across one edge.

/// Filters across the edge whose first "q" sample is `buf[pos]`; `step`
/// is the distance between samples perpendicular to the edge (1 for a
/// vertical edge, the stride for a horizontal one). `filter_size` is the
/// TX_* value of filterSize.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn filter(
    buf: &mut [u16],
    pos: usize,
    step: usize,
    filter_size: u8,
    limit: i32,
    blimit: i32,
    thresh: i32,
    bit_depth: u32,
) {
    let at = |k: isize| -> i32 { buf[(pos as isize + k * step as isize) as usize] as i32 };
    let q0 = at(0);
    let q1 = at(1);
    let q2 = at(2);
    let q3 = at(3);
    let p0 = at(-1);
    let p1 = at(-2);
    let p2 = at(-3);
    let p3 = at(-4);
    let sh = bit_depth - 8;
    // filterMask.
    let limit_bd = limit << sh;
    let blimit_bd = blimit << sh;
    if (p3 - p2).abs() > limit_bd
        || (p2 - p1).abs() > limit_bd
        || (p1 - p0).abs() > limit_bd
        || (q1 - q0).abs() > limit_bd
        || (q2 - q1).abs() > limit_bd
        || (q3 - q2).abs() > limit_bd
        || (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit_bd
    {
        return;
    }
    let thresh_bd = thresh << sh;
    let hev = (p1 - p0).abs() > thresh_bd || (q1 - q0).abs() > thresh_bd;
    let one = 1 << sh;
    let flat = filter_size >= 1
        && (p1 - p0).abs() <= one
        && (q1 - q0).abs() <= one
        && (p2 - p0).abs() <= one
        && (q2 - q0).abs() <= one
        && (p3 - p0).abs() <= one
        && (q3 - q0).abs() <= one;
    if filter_size == 0 || !flat {
        narrow(buf, pos, step, hev, bit_depth, p1, p0, q0, q1);
        return;
    }
    let flat2 = filter_size >= 2 && {
        let mut ok = true;
        for k in 4..8 {
            if (at(-(k as isize) - 1) - p0).abs() > one || (at(k as isize) - q0).abs() > one {
                ok = false;
                break;
            }
        }
        ok
    };
    if filter_size == 1 || !flat2 {
        wide(buf, pos, step, 3);
    } else {
        wide(buf, pos, step, 4);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn narrow(
    buf: &mut [u16],
    pos: usize,
    step: usize,
    hev: bool,
    bit_depth: u32,
    p1: i32,
    p0: i32,
    q0: i32,
    q1: i32,
) {
    let lo = -(1 << (bit_depth - 1));
    let hi = (1 << (bit_depth - 1)) - 1;
    let c = |v: i32| v.clamp(lo, hi);
    let off = 0x80 << (bit_depth - 8);
    let ps1 = p1 - off;
    let ps0 = p0 - off;
    let qs0 = q0 - off;
    let qs1 = q1 - off;
    let mut f = if hev { c(ps1 - qs1) } else { 0 };
    f = c(f + 3 * (qs0 - ps0));
    let f1 = c(f + 4) >> 3;
    let f2 = c(f + 3) >> 3;
    buf[pos] = (c(qs0 - f1) + off) as u16;
    buf[pos - step] = (c(ps0 + f2) + off) as u16;
    if !hev {
        let f = (f1 + 1) >> 1;
        buf[pos + step] = (c(qs1 - f) + off) as u16;
        buf[pos - 2 * step] = (c(ps1 + f) + off) as u16;
    }
}

/// The wide filter (8.8.5.3) with `log2_size` 3 (8 taps) or 4 (16 taps).
#[inline]
fn wide(buf: &mut [u16], pos: usize, step: usize, log2_size: u32) {
    let n = (1isize << (log2_size - 1)) - 1;
    let mut s = [0i32; 16];
    // s[k + 8] = sample at offset k, for k in -(n+1)..=n.
    for k in -(n + 1)..=n {
        s[(k + 8) as usize] = buf[(pos as isize + k * step as isize) as usize] as i32;
    }
    let mut f = [0i32; 16];
    for i in -n..n {
        let mut t = s[(i + 8) as usize];
        for j in -n..=n {
            let p = (i + j).clamp(-(n + 1), n);
            t += s[(p + 8) as usize];
        }
        f[(i + 8) as usize] = (t + (1 << (log2_size - 1))) >> log2_size;
    }
    for i in -n..n {
        buf[(pos as isize + i * step as isize) as usize] = f[(i + 8) as usize] as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_step_is_smoothed_by_the_wide_filter() {
        // A step of 2 across a vertical edge in flat regions: the 16-wide
        // filter applies and the result is monotone.
        let mut b = vec![0u16; 16];
        for (i, v) in b.iter_mut().enumerate() {
            *v = if i < 8 { 100 } else { 101 };
        }
        filter(&mut b, 8, 1, 2, 10, 30, 2, 8);
        assert!(b.windows(2).all(|w| w[0] <= w[1]), "{b:?}");
        assert_eq!(b[0], 100);
        assert_eq!(b[15], 101);
    }

    #[test]
    fn strong_edges_are_left_alone() {
        let mut b = vec![0u16; 16];
        for (i, v) in b.iter_mut().enumerate() {
            *v = if i < 8 { 10 } else { 200 };
        }
        let before = b.clone();
        filter(&mut b, 8, 1, 2, 10, 30, 2, 8);
        assert_eq!(b, before);
    }
}
