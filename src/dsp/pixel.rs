//! Small block operations on samples: adding a constant residual, the
//! compound prediction average, copies.

use super::Level;

/// Adds `v` to the `n` x `n` samples at `dst` (row stride `stride`),
/// clamped to 8 bits: the reconstruction of a DC-only block. `v` is clamped
/// to +-(1 << 16) first, as the transform kernels clamp their residuals.
pub(crate) fn add_const(_level: Level, dst: &mut [u16], stride: usize, n: usize, v: i32) {
    let v = v.clamp(-(1 << 16), 1 << 16);
    for i in 0..n {
        for d in dst[i * stride..i * stride + n].iter_mut() {
            *d = (*d as i32 + v).clamp(0, 255) as u16;
        }
    }
}

/// `dst[i] = (a[i] + b[i] + 1) >> 1` for `w` x `h` samples: `a` and `b`
/// have row stride `w`, `dst` has `stride`. The compound prediction of
/// 8.5.2 (`Round2(preds[0] + preds[1], 1)`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn avg(
    level: Level,
    a: &[u16],
    b: &[u16],
    w: usize,
    h: usize,
    dst: &mut [u16],
    stride: usize,
) {
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Sse41 if w >= 8 => {
            // SAFETY: SSE4.1 (a fortiori SSE2) is present at these levels.
            unsafe { super::x86::avg_sse2(a, b, w, h, dst, stride) }
        }
        #[cfg(target_arch = "aarch64")]
        Level::Neon if w >= 8 => super::neon::avg(a, b, w, h, dst, stride),
        _ => avg_scalar(a, b, w, h, dst, stride),
    }
}

pub(crate) fn avg_scalar(a: &[u16], b: &[u16], w: usize, h: usize, dst: &mut [u16], stride: usize) {
    for i in 0..h {
        let (ra, rb) = (&a[i * w..i * w + w], &b[i * w..i * w + w]);
        for ((d, &x), &y) in dst[i * stride..i * stride + w].iter_mut().zip(ra).zip(rb) {
            *d = ((x as u32 + y as u32 + 1) >> 1) as u16;
        }
    }
}

/// Sum of squared differences of `w` x `h` samples of `a` and `b` (row
/// strides `sa`, `sb`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sse(
    level: Level,
    a: &[u16],
    sa: usize,
    b: &[u16],
    sb: usize,
    w: usize,
    h: usize,
) -> u64 {
    if w == 0 || h == 0 {
        return 0;
    }
    assert!(a.len() >= (h - 1) * sa + w && b.len() >= (h - 1) * sb + w);
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Sse41 if w >= 8 => {
            // SAFETY: SSE4.1 is present; the asserts bound the reads.
            unsafe { super::x86::sse_sse41(a, sa, b, sb, w, h) }
        }
        #[cfg(target_arch = "aarch64")]
        Level::Neon if w >= 8 => super::neon::sse(a, sa, b, sb, w, h),
        _ => sse_scalar(a, sa, b, sb, w, h),
    }
}

pub(crate) fn sse_scalar(a: &[u16], sa: usize, b: &[u16], sb: usize, w: usize, h: usize) -> u64 {
    let mut acc = 0u64;
    for i in 0..h {
        for (&x, &y) in a[i * sa..i * sa + w].iter().zip(&b[i * sb..i * sb + w]) {
            let d = x as i64 - y as i64;
            acc += (d * d) as u64;
        }
    }
    acc
}

/// Sum of absolute differences, as [`sse`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn sad(
    level: Level,
    a: &[u16],
    sa: usize,
    b: &[u16],
    sb: usize,
    w: usize,
    h: usize,
) -> u64 {
    if w == 0 || h == 0 {
        return 0;
    }
    assert!(a.len() >= (h - 1) * sa + w && b.len() >= (h - 1) * sb + w);
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Sse41 if w >= 8 => {
            // SAFETY: SSE4.1 is present; the asserts bound the reads.
            unsafe { super::x86::sad_sse41(a, sa, b, sb, w, h) }
        }
        #[cfg(target_arch = "aarch64")]
        Level::Neon if w >= 8 => super::neon::sad(a, sa, b, sb, w, h),
        _ => sad_scalar(a, sa, b, sb, w, h),
    }
}

pub(crate) fn sad_scalar(a: &[u16], sa: usize, b: &[u16], sb: usize, w: usize, h: usize) -> u64 {
    let mut acc = 0u64;
    for i in 0..h {
        for (&x, &y) in a[i * sa..i * sa + w].iter().zip(&b[i * sb..i * sb + w]) {
            acc += (x as i32 - y as i32).unsigned_abs() as u64;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{Rng, test_levels};

    #[test]
    fn sse_and_sad_match_scalar() {
        let mut rng = Rng(9);
        for &(w, h) in &[(1, 1), (7, 3), (8, 8), (12, 5), (16, 16), (33, 9), (64, 64)] {
            for max in [255, 4095] {
                let (sa, sb) = (w + 5, w + 1);
                let mut a: Vec<u16> = (0..sa * h).map(|_| rng.range(0, max) as u16).collect();
                let mut b: Vec<u16> = (0..sb * h).map(|_| rng.range(0, max) as u16).collect();
                if w == 64 {
                    // Extremes: every difference the largest.
                    a.iter_mut().for_each(|v| *v = max as u16);
                    b.iter_mut().for_each(|v| *v = 0);
                }
                let want = (
                    sse_scalar(&a, sa, &b, sb, w, h),
                    sad_scalar(&a, sa, &b, sb, w, h),
                );
                for l in test_levels() {
                    assert_eq!(sse(l, &a, sa, &b, sb, w, h), want.0, "{l:?} sse {w}x{h}");
                    assert_eq!(sad(l, &a, sa, &b, sb, w, h), want.1, "{l:?} sad {w}x{h}");
                }
            }
        }
    }

    #[test]
    fn avg_matches_scalar() {
        let mut rng = Rng(5);
        for &(w, h) in &[(4, 4), (8, 4), (8, 8), (16, 8), (32, 64), (64, 64), (24, 3)] {
            for max in [255, 1023, 4095, 65535] {
                let a: Vec<u16> = (0..w * h).map(|_| rng.range(0, max) as u16).collect();
                let b: Vec<u16> = (0..w * h).map(|_| rng.range(0, max) as u16).collect();
                let stride = w + 3;
                let mut want = vec![7u16; stride * h];
                avg_scalar(&a, &b, w, h, &mut want, stride);
                for l in test_levels() {
                    let mut got = vec![7u16; stride * h];
                    avg(l, &a, &b, w, h, &mut got, stride);
                    assert_eq!(got, want, "{l:?} {w}x{h}");
                }
            }
        }
    }
}
