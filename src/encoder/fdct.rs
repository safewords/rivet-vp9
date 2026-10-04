//! Forward transforms for the encoder.
//!
//! The decoder's inverse transforms are fixed by the specification; the
//! forward transforms are not. These are the inverses of the real-valued
//! transforms the integer ones approximate: the DCT `y[k] = x[0]/sqrt(2) +
//! sum x[f] cos((2k+1) f pi / 2N)`, the 4-point ADST `y[k] = (2 sqrt(2)/3)
//! sum x[f] sin((2f+1)(k+1) pi / 9)` and the 8 / 16-point ADST `y[k] = sum
//! x[f] sin((2k+1)(2f+1) pi / 4N)` (the decoder's tests check its
//! transforms against these), as matrices computed once in floating point
//! and rounded to 14 fractional bits.
//!
//! The 2D product `D = 2^s Fc R Fr^T` is computed in 16-bit fixed point:
//! the column pass `U = Fc R` is rounded to as many fractional bits as fit
//! 16 bits at the bit depth, the row pass `U Fr^T` is exact in 32 bits.
//! Integer arithmetic makes the result the same on every machine and with
//! every kernel (scalar, SSE4.1, AVX2, NEON: pmaddwd / vmlal products).
//! The lossless Walsh-Hadamard transform is integer and exactly inverts
//! the decoder's.

// Loops index arrays the way the specification's formulas do.
#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::consts::{ADST_ADST, ADST_DCT, DCT_ADST};
use crate::dsp::Level;

/// An N x N matrix, row-major.
type Mat = Vec<f64>;

fn inverse_matrix(m: &Mat, n: usize) -> Mat {
    // Gauss-Jordan with partial pivoting.
    let mut a = m.clone();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&x, &y| a[x * n + col].abs().total_cmp(&a[y * n + col].abs()))
            .unwrap();
        for k in 0..n {
            a.swap(col * n + k, piv * n + k);
            inv.swap(col * n + k, piv * n + k);
        }
        let p = a[col * n + col];
        for k in 0..n {
            a[col * n + k] /= p;
            inv[col * n + k] /= p;
        }
        for r in 0..n {
            if r != col {
                let f = a[r * n + col];
                if f != 0.0 {
                    for k in 0..n {
                        a[r * n + k] -= f * a[col * n + k];
                        inv[r * n + k] -= f * inv[col * n + k];
                    }
                }
            }
        }
    }
    inv
}

/// The inverse 1D transform as a matrix: spatial[k] = sum M[k][f] coef[f].
fn inverse_basis(n: usize, adst: bool) -> Mat {
    let mut m = vec![0.0; n * n];
    for k in 0..n {
        for f in 0..n {
            m[k * n + f] = if !adst {
                let c = if f == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                c * ((2 * k + 1) as f64 * f as f64 * PI / (2 * n) as f64).cos()
            } else if n == 4 {
                (2.0 / 3.0) * 2f64.sqrt() * (((2 * f + 1) * (k + 1)) as f64 * PI / 9.0).sin()
            } else {
                ((2 * k + 1) as f64 * (2 * f + 1) as f64 * PI / (4 * n) as f64).sin()
            };
        }
    }
    m
}

/// Fractional bits of the matrices.
const QBITS: u32 = 14;

/// A forward 1D transform in fixed point.
pub(crate) struct FwdMat {
    pub n0: usize,
    /// `q[f * n0 + k]`: F[f][k] * 2^14, rounded.
    pub q: Vec<i16>,
    /// `qt[k * n0 + f] = q[f * n0 + k]` (the NEON kernel's).
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub qt: Vec<i16>,
    /// `pk[f * n0 / 2 + kp]`: the pair q[f][2kp], q[f][2kp + 1] as one
    /// 32-bit value (low half first), what pmaddwd multiplies (x86-64).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub pk: Vec<i32>,
    /// `cols[kp * n0 + g] = pk[g * n0 / 2 + kp]` (x86-64).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub cols: Vec<i32>,
    /// The largest sum of absolute values of a row of `q`, over 2^14.
    pub l1: f64,
}

fn pack(a: i16, b: i16) -> i32 {
    ((b as u16 as u32) << 16 | a as u16 as u32) as i32
}

impl FwdMat {
    fn new(f: &Mat, n0: usize) -> Self {
        let q: Vec<i16> = f
            .iter()
            .map(|&v| {
                let s = (v * (1u32 << QBITS) as f64).round();
                assert!(s.abs() < 32768.0);
                s as i16
            })
            .collect();
        let mut qt = vec![0i16; n0 * n0];
        for f in 0..n0 {
            for k in 0..n0 {
                qt[k * n0 + f] = q[f * n0 + k];
            }
        }
        let half = n0 / 2;
        let mut pk = vec![0i32; n0 * half];
        for f in 0..n0 {
            for kp in 0..half {
                pk[f * half + kp] = pack(q[f * n0 + 2 * kp], q[f * n0 + 2 * kp + 1]);
            }
        }
        let mut cols = vec![0i32; half * n0];
        for kp in 0..half {
            for g in 0..n0 {
                cols[kp * n0 + g] = pk[g * half + kp];
            }
        }
        let l1 = (0..n0)
            .map(|f| (0..n0).map(|k| (q[f * n0 + k] as f64).abs()).sum::<f64>())
            .fold(0.0, f64::max)
            / (1u32 << QBITS) as f64;
        FwdMat {
            n0,
            q,
            qt,
            pk,
            cols,
            l1,
        }
    }
}

/// Forward 1D matrices, indexed [log2 size - 2][adst].
fn matrix(n_log2: u32, adst: bool) -> &'static FwdMat {
    static CACHE: OnceLock<Vec<FwdMat>> = OnceLock::new();
    let all = CACHE.get_or_init(|| {
        let mut v = Vec::new();
        for l in 2..=5u32 {
            for a in [false, true] {
                let n = 1usize << l;
                v.push(FwdMat::new(&inverse_matrix(&inverse_basis(n, a), n), n));
            }
        }
        v
    });
    &all[((n_log2 - 2) * 2 + adst as u32) as usize]
}

/// The fractional bits the column pass keeps: as many as let `U` (at most
/// the residual's magnitude times the column matrix's row sums) fit 16
/// bits.
fn frac_bits(fc: &FwdMat, bit_depth: u32) -> u32 {
    let bound = fc.l1 * ((1u32 << bit_depth) - 1) as f64 + 1.0;
    let mut p = 0;
    while p < QBITS && bound * (1u32 << (p + 1)) as f64 + 1.0 < 32767.0 {
        p += 1;
    }
    p
}

/// The transform's parameters for a block.
pub(crate) struct Fwd2d {
    pub fc: &'static FwdMat,
    pub fr: &'static FwdMat,
    /// The column pass's rounding shift: 14 - fractional bits kept.
    pub shift1: u32,
}

/// The forward 2D transform of the `1 << n` square residual `res`
/// (row-major), into `out`: the values the decoder calls Dequant (those
/// whose inverse transform gives the residual back), times `2^K`. Returns
/// `K`.
pub(crate) fn forward(
    level: Level,
    res: &[i16],
    n: u32,
    tx_type: u8,
    bit_depth: u32,
    out: &mut [i32],
) -> u32 {
    let n0 = 1usize << n;
    assert!(res.len() >= n0 * n0 && out.len() >= n0 * n0);
    // The decoder: rows use ADST for DCT_ADST and ADST_ADST, columns for
    // ADST_DCT and ADST_ADST.
    let row_adst = tx_type == DCT_ADST || tx_type == ADST_ADST;
    let col_adst = tx_type == ADST_DCT || tx_type == ADST_ADST;
    let fc = matrix(n, col_adst);
    let fr = matrix(n, row_adst);
    let p = frac_bits(fc, bit_depth);
    let t = Fwd2d {
        fc,
        fr,
        shift1: QBITS - p,
    };
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 if n0 >= 16 => {
            // SAFETY: AVX2 is present; the slices hold n0 * n0 values.
            unsafe { crate::dsp::x86::fdct_avx2(&t, res, out) }
        }
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Sse41 => {
            // SAFETY: SSE4.1 is present; as above.
            unsafe { crate::dsp::x86::fdct_sse41(&t, res, out) }
        }
        #[cfg(target_arch = "aarch64")]
        Level::Neon => crate::dsp::neon::fdct(&t, res, out),
        _ => forward_scalar(&t, res, out),
    }
    // D = 2^s Fc R Fr^T, and out = 2^(14 + p) Fc R Fr^T.
    let s = (n + 2).min(6);
    QBITS + p - s
}

/// [`forward`]'s arithmetic, in plain Rust.
pub(crate) fn forward_scalar(t: &Fwd2d, res: &[i16], out: &mut [i32]) {
    let n0 = t.fc.n0;
    let mut u = [0i16; 32 * 32];
    let rnd = 1i32 << (t.shift1 - 1);
    for f in 0..n0 {
        let fc = &t.fc.q[f * n0..f * n0 + n0];
        for j in 0..n0 {
            let mut acc = 0i32;
            for k in 0..n0 {
                acc += fc[k] as i32 * res[k * n0 + j] as i32;
            }
            u[f * n0 + j] = ((acc + rnd) >> t.shift1) as i16;
        }
    }
    for f in 0..n0 {
        let ur = &u[f * n0..f * n0 + n0];
        for g in 0..n0 {
            let fr = &t.fr.q[g * n0..g * n0 + n0];
            let mut acc = 0i32;
            for j in 0..n0 {
                acc += ur[j] as i32 * fr[j] as i32;
            }
            out[f * n0 + g] = acc;
        }
    }
}

/// Quantises the forward transform `d` (scaled by `2^k`) of an `n0`
/// square block with the quantisers `(dc, ac)`: plain rounding with a dead
/// zone (the DC rounds at one half, the rest at 0.38), capped at
/// `max_coef`. `dq_denom` is 2 for 32 x 32 (whose dequantisation halves).
///
/// In single precision, the same operations in the same order in every
/// kernel (convert, multiply, add, truncate: no fused multiply-add), so
/// the result is the same on every machine.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize(
    level: Level,
    d: &[i32],
    k: u32,
    q: (i32, i32),
    dq_denom: u32,
    max_coef: i32,
    out: &mut [i32],
) {
    let unit = dq_denom as f64 / (1u64 << k) as f64;
    let ac = (unit / q.1 as f64) as f32;
    let len = d.len().min(out.len());
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Sse41 if len >= 4 => {
            // SAFETY: SSE4.1 is present; the kernel reads and writes `len`.
            unsafe {
                crate::dsp::x86::quantize_sse41(&d[..len], ac, AC_BIAS, max_coef, &mut out[..len])
            }
        }
        #[cfg(target_arch = "aarch64")]
        Level::Neon if len >= 4 => {
            crate::dsp::neon::quantize(&d[..len], ac, AC_BIAS, max_coef, &mut out[..len])
        }
        _ => quantize_scalar(&d[..len], ac, AC_BIAS, max_coef, &mut out[..len]),
    }
    if len > 0 {
        let dc = (unit / q.0 as f64) as f32;
        out[0] = quantize_one(d[0], dc, DC_BIAS, max_coef);
    }
}

const DC_BIAS: f32 = 0.5;
const AC_BIAS: f32 = 0.38;

#[inline(always)]
fn quantize_one(v: i32, scale: f32, bias: f32, max_coef: i32) -> i32 {
    // |v| < 2^30 (the transform's bound), and the quotient far below 2^31:
    // the conversion truncates, which for a non-negative value is the floor.
    let l = (((v.unsigned_abs() as f32) * scale + bias) as i32).min(max_coef);
    if v < 0 { -l } else { l }
}

/// [`quantize`]'s arithmetic for every coefficient with one scale.
pub(crate) fn quantize_scalar(d: &[i32], scale: f32, bias: f32, max_coef: i32, out: &mut [i32]) {
    for (&v, o) in d.iter().zip(out.iter_mut()) {
        *o = quantize_one(v, scale, bias, max_coef);
    }
}

/// The exact inverse of the decoder's 1D inverse WHT (8.7.1.10) with
/// shift 0: lifting steps undone in reverse order.
fn fwht1(t: &mut [i64; 4]) {
    let (a2, b1, c1, d2) = (t[0], t[1], t[2], t[3]);
    let a1 = a2 + b1;
    let d1 = d2 - c1;
    let e = (a1 - d1) >> 1;
    let b0 = e - b1;
    let c0 = e - c1;
    let a0 = a1 - c0;
    let d0 = d1 + b0;
    // The decoder reads a = T[0], c = T[1], d = T[2], b = T[3].
    *t = [a0, c0, d0, b0];
}

/// Lossless forward transform: integer coefficients whose dequantised
/// values (x4, the lossless quantiser) the decoder's inverse WHT maps back
/// to `residual` exactly.
pub(crate) fn forward_wht(residual: &[i16]) -> [i32; 16] {
    let mut m = [[0i64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = residual[i * 4 + j] as i64;
        }
    }
    // The decoder does rows (shift 2) then columns (shift 0); undo the
    // columns first.
    for j in 0..4 {
        let mut t = [m[0][j], m[1][j], m[2][j], m[3][j]];
        fwht1(&mut t);
        for i in 0..4 {
            m[i][j] = t[i];
        }
    }
    for row in m.iter_mut() {
        fwht1(row);
    }
    let mut out = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            out[i * 4 + j] = m[i][j] as i32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::DCT_DCT;
    use crate::dsp::itx::inverse_transform_2d;
    use crate::dsp::{Rng, test_levels};

    fn lcg(seed: &mut u64) -> i16 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((*seed >> 33) % 511) as i32 - 255) as i16
    }

    #[test]
    fn forward_then_decoder_inverse_round_trips() {
        let mut seed = 3;
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            for tx_type in 0..4u8 {
                if n == 5 && tx_type != DCT_DCT {
                    continue;
                }
                let res: Vec<i16> = (0..n0 * n0).map(|_| lcg(&mut seed)).collect();
                let mut d = vec![0i32; n0 * n0];
                let k = forward(Level::Scalar, &res, n, tx_type, 8, &mut d);
                let mut block: Vec<i32> = d
                    .iter()
                    .map(|&v| (v as f64 / (1u64 << k) as f64).round() as i32)
                    .collect();
                inverse_transform_2d(&mut block, n, tx_type, false);
                let err = block
                    .iter()
                    .zip(&res)
                    .map(|(a, b)| (a - *b as i32).abs())
                    .max()
                    .unwrap();
                assert!(err <= 2, "n={n} type={tx_type}: max error {err}");
            }
        }
    }

    #[test]
    fn matrices_fit_the_fixed_point_bounds() {
        for n in 2..=5u32 {
            for adst in [false, true] {
                let m = matrix(n, adst);
                // Pass 2's 32-bit sums: |U| < 2^15 times the row sums.
                assert!(m.l1 < 2.0, "n={n} adst={adst}: {}", m.l1);
                for bd in [8, 10, 12] {
                    assert!(frac_bits(m, bd) >= 1, "n={n} {bd}-bit");
                }
            }
        }
    }

    #[test]
    fn simd_equals_scalar() {
        let mut rng = Rng(42);
        for bd in [8u32, 10, 12] {
            let max = (1i64 << bd) - 1;
            for n in 2..=5u32 {
                let n0 = 1usize << n;
                for tx_type in 0..4u8 {
                    for iter in 0..50 {
                        let res: Vec<i16> = (0..n0 * n0)
                            .map(|_| match iter % 3 {
                                0 => rng.range(-max, max),
                                1 => [-max, max][rng.range(0, 1) as usize],
                                _ => rng.range(-8, 8),
                            } as i16)
                            .collect();
                        let mut want = vec![0i32; n0 * n0];
                        let k = forward(Level::Scalar, &res, n, tx_type, bd, &mut want);
                        for l in test_levels() {
                            let mut got = vec![0i32; n0 * n0];
                            assert_eq!(forward(l, &res, n, tx_type, bd, &mut got), k);
                            assert_eq!(got, want, "{l:?} {bd}-bit n={n} type={tx_type}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn quantize_simd_equals_scalar() {
        let mut rng = Rng(7);
        for iter in 0..2000 {
            let len = [1usize, 3, 4, 16, 64, 256, 1024][iter % 7];
            let d: Vec<i32> = (0..len)
                .map(|_| match iter % 3 {
                    0 => rng.range(-(1 << 29), 1 << 29),
                    1 => rng.range(-5000, 5000),
                    _ => [-(1 << 30) + 1, (1 << 30) - 1, 0][rng.range(0, 2) as usize],
                } as i32)
                .collect();
            let k = rng.range(8, 20) as u32;
            let q = (rng.range(4, 1828) as i32, rng.range(4, 1828) as i32);
            let dq = rng.range(1, 2) as u32;
            let max_coef = [67 + (1 << 14) - 1, 100][iter % 2];
            let mut want = vec![0i32; len];
            quantize(Level::Scalar, &d, k, q, dq, max_coef, &mut want);
            for l in test_levels() {
                let mut got = vec![0i32; len];
                quantize(l, &d, k, q, dq, max_coef, &mut got);
                assert_eq!(got, want, "{l:?} len {len}");
            }
        }
    }

    #[test]
    fn wht_is_exactly_invertible() {
        let mut seed = 9;
        for _ in 0..1000 {
            let res: Vec<i16> = (0..16).map(|_| lcg(&mut seed)).collect();
            let c = forward_wht(&res);
            let mut block: Vec<i32> = c.iter().map(|&v| v * 4).collect();
            inverse_transform_2d(&mut block, 2, DCT_DCT, true);
            assert_eq!(block, res.iter().map(|&v| v as i32).collect::<Vec<_>>());
        }
    }
}
