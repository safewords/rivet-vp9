//! Forward transforms for the encoder.
//!
//! The decoder's inverse transforms are fixed by the specification; the
//! forward transforms are not, so these are plain floating-point inverses
//! of the real-valued transforms the integer ones approximate: the DCT
//! `y[k] = x[0]/sqrt(2) + sum x[f] cos((2k+1) f pi / 2N)`, the 4-point
//! ADST `y[k] = (2 sqrt(2)/3) sum x[f] sin((2f+1)(k+1) pi / 9)`, and the
//! 8 / 16-point ADST `y[k] = sum x[f] sin((2k+1)(2f+1) pi / 4N)` (the
//! decoder's unit tests check its transforms against these). The lossless
//! Walsh-Hadamard transform is integer and exactly inverts the decoder's.

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::consts::{ADST_ADST, ADST_DCT, DCT_ADST};

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
        let piv = (col..n).max_by(|&x, &y| a[x * n + col].abs().total_cmp(&a[y * n + col].abs())).unwrap();
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

/// Forward 1D matrices, indexed [log2 size - 2][adst].
fn forward(n_log2: u32, adst: bool) -> &'static Mat {
    static CACHE: OnceLock<Vec<Mat>> = OnceLock::new();
    let all = CACHE.get_or_init(|| {
        let mut v = Vec::new();
        for l in 2..=5u32 {
            for a in [false, true] {
                let n = 1usize << l;
                v.push(inverse_matrix(&inverse_basis(n, a), n));
            }
        }
        v
    });
    &all[((n_log2 - 2) * 2 + adst as u32) as usize]
}

/// The forward 2D transform of the `1 << n` square `residual` (row-major),
/// scaled so that the decoder's inverse transform of the result gives the
/// residual back: the values the decoder calls Dequant.
pub(crate) fn forward_2d(residual: &[i32], n: u32, tx_type: u8) -> Vec<f64> {
    let n0 = 1usize << n;
    // The decoder: rows use ADST for DCT_ADST and ADST_ADST, columns for
    // ADST_DCT and ADST_ADST.
    let row_adst = tx_type == DCT_ADST || tx_type == ADST_ADST;
    let col_adst = tx_type == ADST_DCT || tx_type == ADST_ADST;
    let fr = forward(n, row_adst);
    let fc = forward(n, col_adst);
    // residual = Mc . D . Mr^T / 2^s  =>  D = 2^s Fc . residual . Fr^T.
    let shift = (n + 2).min(6);
    let scale = (1u32 << shift) as f64;
    let mut tmp = vec![0.0; n0 * n0];
    // tmp = residual . Fr^T  (each row transformed)
    for i in 0..n0 {
        for f in 0..n0 {
            let mut s = 0.0;
            for k in 0..n0 {
                s += residual[i * n0 + k] as f64 * fr[f * n0 + k];
            }
            tmp[i * n0 + f] = s;
        }
    }
    let mut out = vec![0.0; n0 * n0];
    for f in 0..n0 {
        for j in 0..n0 {
            let mut s = 0.0;
            for k in 0..n0 {
                s += fc[f * n0 + k] * tmp[k * n0 + j];
            }
            out[f * n0 + j] = s * scale;
        }
    }
    out
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
pub(crate) fn forward_wht(residual: &[i32]) -> [i32; 16] {
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

    fn lcg(seed: &mut u64) -> i32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) % 511) as i32 - 255
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
                let res: Vec<i32> = (0..n0 * n0).map(|_| lcg(&mut seed)).collect();
                let d = forward_2d(&res, n, tx_type);
                let mut block: Vec<i32> = d.iter().map(|v| v.round() as i32).collect();
                inverse_transform_2d(&mut block, n, tx_type, false);
                let err = block.iter().zip(&res).map(|(a, b)| (a - b).abs()).max().unwrap();
                assert!(err <= 2, "n={n} type={tx_type}: max error {err}");
            }
        }
    }

    #[test]
    fn wht_is_exactly_invertible() {
        let mut seed = 9;
        for _ in 0..1000 {
            let res: Vec<i32> = (0..16).map(|_| lcg(&mut seed)).collect();
            let c = forward_wht(&res);
            let mut block: Vec<i32> = c.iter().map(|&v| v * 4).collect();
            inverse_transform_2d(&mut block, 2, DCT_DCT, true);
            assert_eq!(block, res);
        }
    }
}
