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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{Rng, test_levels};

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
