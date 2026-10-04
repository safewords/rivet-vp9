//! Sample filtering of the loop filter (8.8.5): the masks, the narrow
//! filter and the wide filters.
//!
//! [`filter`] filters one position across one edge, as the specification
//! writes it. [`filter_edges`] filters the edges of one superblock pass of
//! one plane given each 4-sample run's filter size and strength: the scalar
//! version calls [`filter`]; the SIMD versions filter eight positions at
//! once with per-lane parameters (8 rows of a vertical edge, transposed,
//! or 8 columns of a horizontal one), computing every filter and selecting
//! each lane's — the same values, since every filter's arithmetic fits 16
//! bits (the 16-wide filter's sums, up to 16 x 4095, as unsigned).

// Loops index arrays the way the specification's formulas do.
#![allow(clippy::needless_range_loop)]

use super::Level;

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

/// No filtering for a run: its edge is not filtered, or its level is 0.
pub(crate) const SKIP: i8 = -1;

/// The filtering of one superblock pass of one plane: for each edge (`e`,
/// at most 16, 4 samples apart) and each run of 4 samples along it (`r`,
/// at most 16), the filter size (a TX_* value, or [`SKIP`]) and limit,
/// blimit and thresh (unshifted).
#[derive(Clone)]
pub(crate) struct Edges {
    pub fs: [[i8; 16]; 16],
    pub limit: [[u8; 16]; 16],
    pub blimit: [[u8; 16]; 16],
    pub thresh: [[u8; 16]; 16],
}

impl Edges {
    pub(crate) fn new() -> Self {
        Edges {
            fs: [[SKIP; 16]; 16],
            limit: [[0; 16]; 16],
            blimit: [[0; 16]; 16],
            thresh: [[0; 16]; 16],
        }
    }
}

/// Filters the edges of one pass over the region at (`x0`, `y0`) of `buf`
/// (plane coordinates, row stride `stride`): `vertical` edges are at `x0 +
/// 4e` and runs at rows `y0 + 4r`; horizontal edges at `y0 + 4e` and runs
/// at columns `x0 + 4r`. Edges are filtered in order (left to right, top to
/// bottom), as the specification's loop does; positions along an edge are
/// independent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_edges(
    level: Level,
    buf: &mut [u16],
    stride: usize,
    x0: usize,
    y0: usize,
    vertical: bool,
    n_edges: usize,
    n_runs: usize,
    e: &Edges,
    bit_depth: u32,
) {
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Sse41 if n_runs.is_multiple_of(2) => {
            // SAFETY: SSE4.1 is present at these levels.
            unsafe {
                super::x86::lf_sse41(buf, stride, x0, y0, vertical, n_edges, n_runs, e, bit_depth)
            }
        }
        #[cfg(target_arch = "aarch64")]
        Level::Neon if n_runs.is_multiple_of(2) => edges_simd::<super::neon::V16x8>(
            buf, stride, x0, y0, vertical, n_edges, n_runs, e, bit_depth,
        ),
        _ => edges_scalar(buf, stride, x0, y0, vertical, n_edges, n_runs, e, bit_depth),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn edges_scalar(
    buf: &mut [u16],
    stride: usize,
    x0: usize,
    y0: usize,
    vertical: bool,
    n_edges: usize,
    n_runs: usize,
    e: &Edges,
    bit_depth: u32,
) {
    let (step, along) = if vertical { (1, stride) } else { (stride, 1) };
    for ed in 0..n_edges {
        for r in 0..n_runs {
            let fs = e.fs[ed][r];
            if fs == SKIP {
                continue;
            }
            let pos = if vertical {
                (y0 + 4 * r) * stride + x0 + 4 * ed
            } else {
                (y0 + 4 * ed) * stride + x0 + 4 * r
            };
            for k in 0..4 {
                filter(
                    buf,
                    pos + k * along,
                    step,
                    fs as u8,
                    e.limit[ed][r] as i32,
                    e.blimit[ed][r] as i32,
                    e.thresh[ed][r] as i32,
                    bit_depth,
                );
            }
        }
    }
}

/// Eight 16-bit lanes.
///
/// # Safety
///
/// The SIMD implementations use instructions of their instruction set: the
/// methods may only run where it is present (the dispatcher checks).
pub(crate) trait V16: Copy {
    fn splat(v: i16) -> Self;
    /// Lanes 0-3 `a`, lanes 4-7 `b`.
    fn halves(a: i16, b: i16) -> Self;
    /// Loads 8 samples.
    fn load(src: &[u16]) -> Self;
    /// Stores 8 samples.
    fn store(self, dst: &mut [u16]);
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn max(self, o: Self) -> Self;
    fn min(self, o: Self) -> Self;
    /// `|self - o|` of samples (both non-negative).
    fn absdiff(self, o: Self) -> Self;
    /// All ones where `self > o` (signed).
    fn gt(self, o: Self) -> Self;
    fn and(self, o: Self) -> Self;
    fn or(self, o: Self) -> Self;
    /// `!self & o`.
    fn andnot(self, o: Self) -> Self;
    /// `mask ? a : b`, `mask` all ones or all zeros per lane.
    fn select(mask: Self, a: Self, b: Self) -> Self;
    /// Arithmetic shift right.
    fn sra(self, n: u32) -> Self;
    /// Logical shift right.
    fn srl(self, n: u32) -> Self;
    /// Whether any lane is nonzero.
    fn any(self) -> bool;
    /// Transposes the 8 x 8 matrix whose rows are `v`.
    fn transpose(v: &mut [Self; 8]);
}

/// The lane parameters of one group of two runs.
#[derive(Clone, Copy)]
struct LaneParams<V> {
    fs: V,
    limit: V,
    blimit: V,
    thresh: V,
    /// Whether any lane uses the 16-wide filter (needs samples 8 away).
    wide16: bool,
}

#[inline(always)]
fn params<V: V16>(e: &Edges, ed: usize, r: usize, sh: u32) -> Option<LaneParams<V>> {
    let (a, b) = (e.fs[ed][r], e.fs[ed][r + 1]);
    if a == SKIP && b == SKIP {
        return None;
    }
    let p = |t: &[[u8; 16]; 16]| V::halves((t[ed][r] as i16) << sh, (t[ed][r + 1] as i16) << sh);
    Some(LaneParams {
        fs: V::halves(a as i16, b as i16),
        limit: p(&e.limit),
        blimit: p(&e.blimit),
        thresh: p(&e.thresh),
        wide16: a == 2 || b == 2,
    })
}

/// The filter on eight positions: `s[8 + k]` holds the samples at offset
/// `k` (-8..8) from the edge, `s[0..4]` and `s[12..16]` only when
/// `p.wide16`. Returns whether anything was filtered.
#[inline(always)]
fn filter_lanes<V: V16>(s: &mut [V; 16], p: &LaneParams<V>, bit_depth: u32) -> bool {
    let sh = bit_depth - 8;
    let (p3, p2, p1, p0) = (s[4], s[5], s[6], s[7]);
    let (q0, q1, q2, q3) = (s[8], s[9], s[10], s[11]);
    let d_p1p0 = p1.absdiff(p0);
    let d_q1q0 = q1.absdiff(q0);
    let dmax = p3
        .absdiff(p2)
        .max(p2.absdiff(p1))
        .max(d_p1p0)
        .max(d_q1q0)
        .max(q2.absdiff(q1))
        .max(q3.absdiff(q2));
    let edge = p0
        .absdiff(q0)
        .add(p0.absdiff(q0))
        .add(p1.absdiff(q1).srl(1));
    let skip = dmax
        .gt(p.limit)
        .or(edge.gt(p.blimit))
        .or(V::splat(0).gt(p.fs));
    // mask: lanes filtered at all.
    let mask = skip.andnot(V::splat(-1));
    if !mask.any() {
        return false;
    }
    let hev = d_p1p0.gt(p.thresh).or(d_q1q0.gt(p.thresh));
    let one = V::splat(1 << sh);
    let fdmax = d_p1p0
        .max(d_q1q0)
        .max(p2.absdiff(p0))
        .max(q2.absdiff(q0))
        .max(p3.absdiff(p0))
        .max(q3.absdiff(q0));
    // flat: filtered, fs >= 1, every difference within one.
    let flat = fdmax.gt(one).andnot(mask.and(p.fs.gt(V::splat(0))));
    let flat2 = if p.wide16 {
        let mut d = s[3].absdiff(p0);
        for k in 0..3 {
            d = d.max(s[k].absdiff(p0));
        }
        for k in 12..16 {
            d = d.max(s[k].absdiff(q0));
        }
        d.gt(one).andnot(flat.and(p.fs.gt(V::splat(1))))
    } else {
        V::splat(0)
    };
    // The narrow filter, on every lane.
    let off = V::splat(0x80 << sh);
    let lo = V::splat(-(1 << (bit_depth - 1)));
    let hi = V::splat((1 << (bit_depth - 1)) - 1);
    let c = |v: V| v.max(lo).min(hi);
    let ps1 = p1.sub(off);
    let ps0 = p0.sub(off);
    let qs0 = q0.sub(off);
    let qs1 = q1.sub(off);
    let f = V::select(hev, c(ps1.sub(qs1)), V::splat(0));
    let d = qs0.sub(ps0);
    let f = c(f.add(d).add(d).add(d));
    let f1 = c(f.add(V::splat(4))).sra(3);
    let f2 = c(f.add(V::splat(3))).sra(3);
    let nq0 = c(qs0.sub(f1)).add(off);
    let np0 = c(ps0.add(f2)).add(off);
    let f3 = f1.add(V::splat(1)).sra(1);
    let nq1 = c(qs1.sub(f3)).add(off);
    let np1 = c(ps1.add(f3)).add(off);
    let nar = flat.andnot(mask);
    let nar1 = hev.andnot(nar);
    // The 8-tap filter: outputs at offsets -3..3, each the rounded mean of
    // the sample and the 7 around it (clamped to -4..3), sliding.
    let at8 = |k: isize| s[(k.clamp(-4, 3) + 8) as usize];
    let mut w8 = [V::splat(0); 6];
    {
        let mut sum = V::splat(0);
        for j in -3..=3isize {
            sum = sum.add(at8(-3 + j));
        }
        for i in -3..3isize {
            if i > -3 {
                sum = sum.sub(at8(i - 4)).add(at8(i + 3));
            }
            w8[(i + 3) as usize] = sum.add(at8(i)).add(V::splat(4)).srl(3);
        }
    }
    // The 16-wide filter: offsets -7..7, sums of 16 samples as unsigned.
    let mut w16 = [V::splat(0); 14];
    if p.wide16 {
        let at16 = |k: isize| s[(k.clamp(-8, 7) + 8) as usize];
        let mut sum = V::splat(0);
        for j in -7..=7isize {
            sum = sum.add(at16(-7 + j));
        }
        for i in -7..7isize {
            if i > -7 {
                sum = sum.sub(at16(i - 8)).add(at16(i + 7));
            }
            w16[(i + 7) as usize] = sum.add(at16(i)).add(V::splat(8)).srl(4);
        }
    }
    let w8_lanes = flat2.andnot(flat);
    for k in -7..7isize {
        let idx = (k + 8) as usize;
        let mut v = s[idx];
        if (-2..2).contains(&k) {
            let n = match k {
                -2 => np1,
                -1 => np0,
                0 => nq0,
                _ => nq1,
            };
            let m = if k == -1 || k == 0 { nar } else { nar1 };
            v = V::select(m, n, v);
        }
        if (-3..3).contains(&k) {
            v = V::select(w8_lanes, w8[(k + 3) as usize], v);
        }
        if p.wide16 {
            v = V::select(flat2, w16[(k + 7) as usize], v);
        }
        s[idx] = v;
    }
    true
}

/// [`edges_scalar`] eight positions at a time: two runs per group.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn edges_simd<V: V16>(
    buf: &mut [u16],
    stride: usize,
    x0: usize,
    y0: usize,
    vertical: bool,
    n_edges: usize,
    n_runs: usize,
    e: &Edges,
    bit_depth: u32,
) {
    let sh = bit_depth - 8;
    let mut s = [V::splat(0); 16];
    if vertical {
        for g in (0..n_runs).step_by(2) {
            let y = y0 + 4 * g;
            for ed in 0..n_edges {
                let Some(p) = params::<V>(e, ed, g, sh) else {
                    continue;
                };
                let x = x0 + 4 * ed;
                // Rows y..y+8, columns x-8..x+8 (or x-4..x+4), transposed.
                let lo = if p.wide16 { 0 } else { 1 };
                for half in lo..2 {
                    let mut t = [V::splat(0); 8];
                    let col = x + 8 * half - 8;
                    // Narrow: the 8 columns x-4..x+4 in one transpose.
                    let col = if p.wide16 { col } else { x - 4 };
                    for r in 0..8 {
                        t[r] = V::load(&buf[(y + r) * stride + col..]);
                    }
                    V::transpose(&mut t);
                    let base = if p.wide16 { 8 * half } else { 4 };
                    s[base..base + 8].copy_from_slice(&t);
                }
                if !filter_lanes(&mut s, &p, bit_depth) {
                    continue;
                }
                for half in lo..2 {
                    let base = if p.wide16 { 8 * half } else { 4 };
                    let col = if p.wide16 { x + 8 * half - 8 } else { x - 4 };
                    let mut t = [V::splat(0); 8];
                    t.copy_from_slice(&s[base..base + 8]);
                    V::transpose(&mut t);
                    for r in 0..8 {
                        t[r].store(&mut buf[(y + r) * stride + col..]);
                    }
                }
            }
        }
    } else {
        for ed in 0..n_edges {
            let y = y0 + 4 * ed;
            for g in (0..n_runs).step_by(2) {
                let Some(p) = params::<V>(e, ed, g, sh) else {
                    continue;
                };
                let x = x0 + 4 * g;
                let (a, b) = if p.wide16 { (0, 16) } else { (4, 12) };
                for k in a..b {
                    s[k] = V::load(&buf[(y + k - 8) * stride + x..]);
                }
                if !filter_lanes(&mut s, &p, bit_depth) {
                    continue;
                }
                let (a, b) = if p.wide16 { (1, 15) } else { (5, 11) };
                for k in a..b {
                    s[k].store(&mut buf[(y + k - 8) * stride + x..]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{Rng, test_levels};

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

    /// Pictures that exercise every branch: flat areas with small steps
    /// (the wide filters), noise (the masks), large steps (hev), extremes.
    fn picture(rng: &mut Rng, w: usize, h: usize, max: i64, kind: usize) -> Vec<u16> {
        let base = rng.range(0, max);
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let v = match kind {
                    0 => base + ((x / 8 + y / 8) % 2) as i64 * rng.range(0, 2),
                    1 => base + rng.range(-3, 3) * (max / 255).max(1),
                    2 => rng.range(0, max),
                    3 => {
                        if (x / 4 + y / 4) % 2 == 0 {
                            0
                        } else {
                            max
                        }
                    }
                    _ => base + ((x + y) % 5) as i64 * rng.range(0, 4),
                };
                v.clamp(0, max) as u16
            })
            .collect()
    }

    #[test]
    fn simd_equals_scalar() {
        let mut rng = Rng(0xabcdef);
        let levels = test_levels();
        for bd in [8u32, 10, 12] {
            let max = (1i64 << bd) - 1;
            for iter in 0..400 {
                let (w, h) = (96, 96);
                let pic = picture(&mut rng, w, h, max, iter % 5);
                let mut e = Edges::new();
                for ed in 0..16 {
                    for r in 0..16 {
                        e.fs[ed][r] = rng.range(-1, 2) as i8;
                        // Filter size 2 only where the specification can
                        // have it: edges 16 samples apart.
                        if e.fs[ed][r] == 2 && ed % 4 != 0 {
                            e.fs[ed][r] = 1;
                        }
                        let l = rng.range(1, 63) as i32;
                        let limit = (l >> rng.range(0, 2)).clamp(1, 9);
                        e.limit[ed][r] = limit as u8;
                        e.blimit[ed][r] = (2 * (l + 2) + limit) as u8;
                        e.thresh[ed][r] = (l >> 4) as u8;
                    }
                }
                let n_runs = if iter % 3 == 0 { 8 } else { 16 };
                let n_edges = if iter % 4 == 0 { 8 } else { 16 };
                for vertical in [true, false] {
                    let mut want = pic.clone();
                    edges_scalar(&mut want, w, 16, 16, vertical, n_edges, n_runs, &e, bd);
                    for &l in &levels {
                        let mut got = pic.clone();
                        filter_edges(l, &mut got, w, 16, 16, vertical, n_edges, n_runs, &e, bd);
                        if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
                            panic!(
                                "{l:?} {bd}-bit vertical {vertical} iter {iter}: first difference at ({}, {}): {} vs {}",
                                i % w,
                                i / w,
                                got[i],
                                want[i]
                            );
                        }
                    }
                }
            }
        }
    }
}
