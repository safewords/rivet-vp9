//! Block inter prediction (8.5.2.4): the separable 8-tap subpixel filter
//! over a reference plane, with the reference's edge samples repeated
//! outward (the clamps of the specification).
//!
//! An unscaled prediction is one horizontal pass over `h + 7` rows of the
//! reference (each output the filter over 8 samples, rounded and clamped)
//! and one vertical pass over its output: [`h_pass`] and [`v_pass`], which
//! have SIMD versions. A footprint that crosses the reference's edge is
//! first copied out with the edge samples repeated, which is what the
//! specification's clamped coordinates read. Scaled prediction (a
//! reference of another size) follows the specification's formula
//! directly.

// Loops index arrays the way the specification's formulas do.
#![allow(clippy::needless_range_loop)]

use super::Level;
use crate::tables::SUBPEL_FILTERS;

/// A reference plane as the predictor sees it.
pub(crate) struct RefPlane<'a> {
    pub data: &'a [u16],
    pub stride: usize,
    /// lastX / lastY: the bottom-right sample of the plane.
    pub last_x: i32,
    pub last_y: i32,
}

/// Working memory of [`predict`], kept between calls.
pub(crate) struct Scratch {
    /// The horizontal pass's output: up to 71 rows of 64.
    tmp: Vec<u16>,
    /// A footprint copied out at the reference's edge: up to 71 x 71.
    edge: Vec<u16>,
}

impl Scratch {
    pub(crate) fn new() -> Self {
        Scratch {
            tmp: vec![0; 71 * 64],
            edge: vec![0; 71 * 71],
        }
    }
}

/// `filter`'s taps for position `frac` as 16-bit values.
#[inline]
pub(crate) fn taps(filter: u8, frac: usize) -> [i16; 8] {
    SUBPEL_FILTERS[filter as usize][frac].map(|v| v as i16)
}

/// The horizontal pass: `dst[r][c]` (row stride `dst_stride`) is the
/// filter over `src[r][c..c + 8]` (row stride `src_stride`), rounded and
/// clamped to `0..=max`, for `rows` x `w`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn h_pass_scalar(
    src: &[u16],
    src_stride: usize,
    rows: usize,
    w: usize,
    f: &[i16; 8],
    max: i32,
    dst: &mut [u16],
    dst_stride: usize,
) {
    for r in 0..rows {
        let s = &src[r * src_stride..r * src_stride + w + 7];
        let d = &mut dst[r * dst_stride..r * dst_stride + w];
        for c in 0..w {
            let mut sum = 0;
            for t in 0..8 {
                sum += f[t] as i32 * s[c + t] as i32;
            }
            d[c] = ((sum + 64) >> 7).clamp(0, max) as u16;
        }
    }
}

/// The vertical pass: `dst[r][c]` is the filter over `src[r..r + 8][c]`,
/// rounded and clamped, for `h` x `w`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn v_pass_scalar(
    src: &[u16],
    src_stride: usize,
    h: usize,
    w: usize,
    f: &[i16; 8],
    max: i32,
    dst: &mut [u16],
    dst_stride: usize,
) {
    for r in 0..h {
        let d = &mut dst[r * dst_stride..r * dst_stride + w];
        for c in 0..w {
            let mut sum = 0;
            for t in 0..8 {
                sum += f[t] as i32 * src[(r + t) * src_stride + c] as i32;
            }
            d[c] = ((sum + 64) >> 7).clamp(0, max) as u16;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn h_pass(
    level: Level,
    src: &[u16],
    src_stride: usize,
    rows: usize,
    w: usize,
    f: &[i16; 8],
    max: i32,
    dst: &mut [u16],
    dst_stride: usize,
) {
    assert!(rows > 0 && src.len() >= (rows - 1) * src_stride + w + 7);
    assert!(dst.len() >= (rows - 1) * dst_stride + w);
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 if w >= 16 => {
            // SAFETY: AVX2 is present; the slices cover the footprint
            // (asserted above), which is all the kernel reads and writes.
            unsafe { super::x86::h_pass_avx2(src, src_stride, rows, w, f, max, dst, dst_stride) }
        }
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Sse41 => {
            // SAFETY: as above, with SSE4.1.
            unsafe { super::x86::h_pass_sse41(src, src_stride, rows, w, f, max, dst, dst_stride) }
        }
        #[cfg(target_arch = "aarch64")]
        Level::Neon => super::neon::h_pass(src, src_stride, rows, w, f, max, dst, dst_stride),
        _ => h_pass_scalar(src, src_stride, rows, w, f, max, dst, dst_stride),
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn v_pass(
    level: Level,
    src: &[u16],
    src_stride: usize,
    h: usize,
    w: usize,
    f: &[i16; 8],
    max: i32,
    dst: &mut [u16],
    dst_stride: usize,
) {
    assert!(h > 0 && src.len() >= (h + 6) * src_stride + w);
    assert!(dst.len() >= (h - 1) * dst_stride + w);
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 if w >= 16 => {
            // SAFETY: AVX2 is present; the slices cover the footprint.
            unsafe { super::x86::v_pass_avx2(src, src_stride, h, w, f, max, dst, dst_stride) }
        }
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Sse41 => {
            // SAFETY: as above, with SSE4.1.
            unsafe { super::x86::v_pass_sse41(src, src_stride, h, w, f, max, dst, dst_stride) }
        }
        #[cfg(target_arch = "aarch64")]
        Level::Neon => super::neon::v_pass(src, src_stride, h, w, f, max, dst, dst_stride),
        _ => v_pass_scalar(src, src_stride, h, w, f, max, dst, dst_stride),
    }
}

/// The unscaled prediction from the `w + 7` x `h + 7` footprint at `src`
/// (row stride `stride`; its first sample is 3 rows above and 3 columns
/// left of the block's integer position).
#[allow(clippy::too_many_arguments)]
fn unscaled(
    level: Level,
    tmp: &mut [u16],
    src: &[u16],
    stride: usize,
    fx: usize,
    fy: usize,
    filter: u8,
    w: usize,
    h: usize,
    max: i32,
    out: &mut [u16],
    out_stride: usize,
) {
    match (fx, fy) {
        (0, 0) => {
            for r in 0..h {
                let s = (r + 3) * stride + 3;
                out[r * out_stride..r * out_stride + w].copy_from_slice(&src[s..s + w]);
            }
        }
        (_, 0) => h_pass(
            level,
            &src[3 * stride..],
            stride,
            h,
            w,
            &taps(filter, fx),
            max,
            out,
            out_stride,
        ),
        (0, _) => v_pass(
            level,
            &src[3..],
            stride,
            h,
            w,
            &taps(filter, fy),
            max,
            out,
            out_stride,
        ),
        _ => {
            h_pass(level, src, stride, h + 7, w, &taps(filter, fx), max, tmp, w);
            v_pass(level, tmp, w, h, w, &taps(filter, fy), max, out, out_stride);
        }
    }
}

/// Predicts a `w` x `h` block whose top-left is at (`x`, `y`) in 1/16
/// sample units of the reference, stepping `x_step` / `y_step` sixteenths
/// per output sample. Writes `w * h` samples to `out` (row stride
/// `out_stride`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict(
    level: Level,
    scratch: &mut Scratch,
    r: &RefPlane,
    x: i32,
    y: i32,
    x_step: i32,
    y_step: i32,
    w: usize,
    h: usize,
    filter: u8,
    bit_depth: u32,
    out: &mut [u16],
    out_stride: usize,
) {
    let max = (1i32 << bit_depth) - 1;
    let y0 = (y >> 4) - 3;
    let x0 = (x >> 4) - 3;
    if x_step == 16 && y_step == 16 && w <= 64 && h <= 64 {
        let (fx, fy) = ((x & 15) as usize, (y & 15) as usize);
        let (fw, fh) = (w + 7, h + 7);
        if x0 >= 0 && y0 >= 0 && x0 + fw as i32 - 1 <= r.last_x && y0 + fh as i32 - 1 <= r.last_y {
            let base = y0 as usize * r.stride + x0 as usize;
            unscaled(
                level,
                &mut scratch.tmp,
                &r.data[base..],
                r.stride,
                fx,
                fy,
                filter,
                w,
                h,
                max,
                out,
                out_stride,
            );
        } else {
            // The footprint with the edge repeated: what the clamped
            // coordinates of the specification read.
            for row in 0..fh {
                let ry = (y0 + row as i32).clamp(0, r.last_y) as usize;
                let src = &r.data[ry * r.stride..];
                let e = &mut scratch.edge[row * fw..row * fw + fw];
                for (c, v) in e.iter_mut().enumerate() {
                    *v = src[(x0 + c as i32).clamp(0, r.last_x) as usize];
                }
            }
            unscaled(
                level,
                &mut scratch.tmp,
                &scratch.edge,
                fw,
                fx,
                fy,
                filter,
                w,
                h,
                max,
                out,
                out_stride,
            );
        }
        return;
    }
    let taps = &SUBPEL_FILTERS[filter as usize];
    let ih = ((((h as i32 - 1) * y_step + 15) >> 4) + 8) as usize;
    let mut inter = vec![0i32; ih * w];
    // The general case, exactly as written.
    for row in 0..ih {
        let ry = (y0 + row as i32).clamp(0, r.last_y) as usize;
        let src = &r.data[ry * r.stride..];
        for c in 0..w {
            let p = x + x_step * c as i32;
            let f = &taps[(p & 15) as usize];
            let px = (p >> 4) - 3;
            let mut sum = 0;
            for t in 0..8 {
                let sx = (px + t as i32).clamp(0, r.last_x) as usize;
                sum += f[t] * src[sx] as i32;
            }
            inter[row * w + c] = ((sum + 64) >> 7).clamp(0, max);
        }
    }
    for rr in 0..h {
        let p = (y & 15) + y_step * rr as i32;
        let f = &taps[(p & 15) as usize];
        let base = (p >> 4) as usize;
        for c in 0..w {
            let mut sum = 0;
            for t in 0..8 {
                sum += f[t] * inter[(base + t) * w + c];
            }
            out[rr * out_stride + c] = ((sum + 64) >> 7).clamp(0, max) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{Rng, test_levels};

    fn plane() -> Vec<u16> {
        (0..32 * 32).map(|i| ((i * 37) % 251) as u16).collect()
    }

    /// The specification's formula with its clamps, for any step.
    #[allow(clippy::too_many_arguments)]
    fn formula(
        d: &[u16],
        stride: usize,
        last: (i32, i32),
        x: i32,
        y: i32,
        step: (i32, i32),
        w: usize,
        h: usize,
        filter: u8,
        max: i32,
    ) -> Vec<u16> {
        let ih = ((((h as i32 - 1) * step.1 + 15) >> 4) + 8) as usize;
        let mut inter = vec![0i32; ih * w];
        for rr in 0..ih {
            for c in 0..w {
                let p = x + step.0 * c as i32;
                let mut s = 0;
                for t in 0..8 {
                    let sy = ((y >> 4) + rr as i32 - 3).clamp(0, last.1) as usize;
                    let sx = ((p >> 4) + t as i32 - 3).clamp(0, last.0) as usize;
                    s += SUBPEL_FILTERS[filter as usize][(p & 15) as usize][t]
                        * d[sy * stride + sx] as i32;
                }
                inter[rr * w + c] = ((s + 64) >> 7).clamp(0, max);
            }
        }
        let mut b = vec![0u16; w * h];
        for rr in 0..h {
            for c in 0..w {
                let p = (y & 15) + step.1 * rr as i32;
                let mut s = 0;
                for t in 0..8 {
                    s += SUBPEL_FILTERS[filter as usize][(p & 15) as usize][t]
                        * inter[((p >> 4) as usize + t) * w + c];
                }
                b[rr * w + c] = ((s + 64) >> 7).clamp(0, max) as u16;
            }
        }
        b
    }

    #[test]
    fn integer_positions_copy() {
        let d = plane();
        let r = RefPlane {
            data: &d,
            stride: 32,
            last_x: 31,
            last_y: 31,
        };
        let mut out = vec![0u16; 64];
        let mut s = Scratch::new();
        predict(
            Level::Scalar,
            &mut s,
            &r,
            5 * 16,
            7 * 16,
            16,
            16,
            8,
            8,
            0,
            8,
            &mut out,
            8,
        );
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out[y * 8 + x], d[(7 + y) * 32 + 5 + x]);
            }
        }
    }

    /// Every level, every block size, positions inside, across and far
    /// outside the reference's edges, every filter and phase, every bit
    /// depth, and scaled steps: equal to the specification's formula.
    #[test]
    fn every_level_matches_the_formula() {
        let mut rng = Rng(0xfeed);
        let mut levels = vec![Level::Scalar];
        levels.extend(test_levels());
        let mut s = Scratch::new();
        for bd in [8u32, 10, 12] {
            let max = (1i32 << bd) - 1;
            let (pw, ph) = (150usize, 90usize);
            let stride = pw + 5;
            let d: Vec<u16> = (0..stride * ph)
                .map(|i| {
                    if i % 97 == 0 {
                        max as u16
                    } else {
                        rng.range(0, max as i64) as u16
                    }
                })
                .collect();
            let last = (pw as i32 - 1, ph as i32 - 1);
            let r = RefPlane {
                data: &d,
                stride,
                last_x: last.0,
                last_y: last.1,
            };
            for &(w, h) in &[
                (4, 4),
                (4, 8),
                (8, 4),
                (8, 8),
                (16, 8),
                (8, 16),
                (16, 16),
                (32, 16),
                (16, 32),
                (32, 32),
                (64, 32),
                (32, 64),
                (64, 64),
            ] {
                for iter in 0..24 {
                    let filter = rng.range(0, 3) as u8;
                    let x = match iter % 3 {
                        0 => rng.range(0, (pw as i64 - w as i64 - 8) * 16),
                        1 => rng.range(-80 * 16, (pw as i64 + 10) * 16),
                        _ => rng.range(-3 * 16, 4 * 16),
                    } as i32;
                    let y = rng.range(-40 * 16, (ph as i64 + 10) * 16) as i32;
                    let y = if iter % 2 == 0 {
                        rng.range(3 * 16, ((ph - h) as i64 - 5) * 16) as i32
                    } else {
                        y
                    };
                    let step = if iter % 8 == 7 {
                        (rng.range(8, 32) as i32, rng.range(8, 32) as i32)
                    } else {
                        (16, 16)
                    };
                    let want = formula(&d, stride, last, x, y, step, w, h, filter, max);
                    for &l in &levels {
                        let out_stride = w + 3;
                        let mut out = vec![0u16; out_stride * h];
                        predict(
                            l, &mut s, &r, x, y, step.0, step.1, w, h, filter, bd, &mut out,
                            out_stride,
                        );
                        for rr in 0..h {
                            assert_eq!(
                                &out[rr * out_stride..rr * out_stride + w],
                                &want[rr * w..rr * w + w],
                                "{l:?} {bd}-bit {w}x{h} at ({x}, {y}) step {step:?} filter {filter} row {rr}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn filters_sum_to_128() {
        for f in SUBPEL_FILTERS.iter() {
            for phase in f.iter() {
                assert_eq!(phase.iter().sum::<i32>(), 128);
            }
        }
    }
}
