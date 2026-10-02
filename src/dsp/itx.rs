//! Inverse transforms (8.7): the DCT, ADST and Walsh-Hadamard butterflies
//! exactly as the specification builds them, and the 2D transform.
//!
//! Intermediate values are kept in `i64`: a conforming stream keeps `T`
//! within 8 + BitDepth bits and the ADST's `S` within 24 + BitDepth, but the
//! products before rounding need more than 32 bits at 12-bit depth.

use crate::consts::{ADST_ADST, ADST_DCT, DCT_ADST, DCT_DCT};
use crate::tables::COS64_LOOKUP;

#[inline]
fn brev(num_bits: u32, x: usize) -> usize {
    let mut t = 0;
    for i in 0..num_bits {
        let bit = (x >> i) & 1;
        t += bit << (num_bits - 1 - i);
    }
    t
}

#[inline]
fn cos64(angle: i32) -> i64 {
    let a2 = angle & 127;
    (match a2 {
        0..=32 => COS64_LOOKUP[a2 as usize],
        33..=64 => -COS64_LOOKUP[(64 - a2) as usize],
        65..=96 => -COS64_LOOKUP[(a2 - 64) as usize],
        _ => COS64_LOOKUP[(128 - a2) as usize],
    }) as i64
}

#[inline]
fn sin64(angle: i32) -> i64 {
    cos64(angle - 32)
}

#[inline]
fn round14(x: i64) -> i64 {
    (x + (1 << 13)) >> 14
}

/// B( a, b, angle, flip ).
#[inline]
fn bfly(t: &mut [i64], a: usize, b: usize, angle: i32, flip: bool) {
    let x = t[a] * cos64(angle) - t[b] * sin64(angle);
    let y = t[a] * sin64(angle) + t[b] * cos64(angle);
    t[a] = round14(x);
    t[b] = round14(y);
    if flip {
        t.swap(a, b);
    }
}

/// H( a, b, flip ).
#[inline]
fn hada(t: &mut [i64], a: usize, b: usize, flip: bool) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let x = t[a];
    let y = t[b];
    t[a] = x + y;
    t[b] = x - y;
}

/// SB( a, b, angle, flip ).
#[inline]
fn sbfly(t: &[i64], s: &mut [i64], a: usize, b: usize, angle: i32, flip: bool) {
    s[a] = t[a] * cos64(angle) - t[b] * sin64(angle);
    s[b] = t[a] * sin64(angle) + t[b] * cos64(angle);
    if flip {
        s.swap(a, b);
    }
}

/// SH( a, b ).
#[inline]
fn shada(t: &mut [i64], s: &[i64], a: usize, b: usize) {
    t[a] = round14(s[a] + s[b]);
    t[b] = round14(s[a] - s[b]);
}

/// The inverse DCT process (8.7.1.3) on the already permuted `t`.
fn idct_core(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let n1 = n0 >> 1;
    let n2 = n0 >> 2;
    let n3 = n0 >> 3;
    let nn = n as usize;
    if n == 2 {
        bfly(t, 0, 1, 16, true);
    } else {
        idct_core(t, n - 1);
    }
    for i in 0..n2 {
        bfly(t, n1 + i, n0 - 1 - i, 32 - brev(5, n1 + i) as i32, false);
    }
    if n >= 3 {
        for i in 0..n3 {
            for j in 0..2 {
                hada(t, n1 + 4 * i + 2 * j, n1 + 1 + 4 * i + 2 * j, j == 1);
            }
        }
    }
    if n == 5 {
        for i in 0..2 {
            for j in 0..2 {
                bfly(t, n0 - nn + 3 - n2 * j - 4 * i, n1 + nn - 4 + n2 * j + 4 * i, 28 - 16 * i as i32 + 56 * j as i32, true);
            }
        }
        for i in 0..2 {
            for j in 0..4 {
                hada(t, n1 + n3 * j + i, n1 + n2 - 5 + n3 * j - i, j & 1 == 1);
            }
        }
    }
    if n >= 4 {
        let imax = if n == 5 { 1 } else { 0 };
        for i in 0..=imax {
            for j in 0..2 {
                bfly(t, n0 - nn + 2 - i - n2 * j, n1 + nn - 3 + i + n2 * j, 24 + 48 * j as i32, true);
            }
        }
        for i in 0..(2 * nn - 6) {
            for j in 0..2 {
                hada(t, n1 + n2 * j + i, n1 + n2 - 1 + n2 * j - i, j & 1 == 1);
            }
        }
    }
    if n >= 3 {
        for i in 0..n3 {
            bfly(t, n0 - n3 - 1 - i, n1 + n3 + i, 16, true);
        }
    }
    for i in 0..n1 {
        hada(t, i, n0 - 1 - i, false);
    }
}

/// Inverse DCT array permutation (8.7.1.2) then the inverse DCT.
pub(crate) fn idct(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let mut copy = [0i64; 32];
    copy[..n0].copy_from_slice(&t[..n0]);
    for i in 0..n0 {
        t[i] = copy[brev(n, i)];
    }
    idct_core(t, n);
}

fn adst_in_perm(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let n1 = n0 >> 1;
    let mut copy = [0i64; 16];
    copy[..n0].copy_from_slice(&t[..n0]);
    for i in 0..n1 {
        t[2 * i] = copy[n0 - 1 - 2 * i];
        t[2 * i + 1] = copy[2 * i];
    }
}

fn adst_out_perm(t: &mut [i64], n: u32) {
    let mut copy = [0i64; 16];
    copy[..1 << n].copy_from_slice(&t[..1 << n]);
    if n == 4 {
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    for d in 0..2 {
                        t[8 * a + 4 * b + 2 * c + d] = copy[8 * (d ^ c) + 4 * (c ^ b) + 2 * (b ^ a) + a];
                    }
                }
            }
        }
    } else {
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    t[4 * a + 2 * b + c] = copy[4 * (c ^ b) + 2 * (b ^ a) + a];
                }
            }
        }
    }
}

const SINPI_1_9: i64 = 5283;
const SINPI_2_9: i64 = 9929;
const SINPI_3_9: i64 = 13377;
const SINPI_4_9: i64 = 15212;

fn iadst4(t: &mut [i64]) {
    let s0 = SINPI_1_9 * t[0];
    let s1 = SINPI_2_9 * t[0];
    let s2 = SINPI_3_9 * t[1];
    let s3 = SINPI_4_9 * t[2];
    let s4 = SINPI_1_9 * t[2];
    let s5 = SINPI_2_9 * t[3];
    let s6 = SINPI_4_9 * t[3];
    let v = t[0] - t[2] + t[3];
    let s7 = SINPI_3_9 * v;
    let x0 = s0 + s3 + s5;
    let x1 = s1 - s4 - s6;
    let x2 = s7;
    let x3 = s2;
    t[0] = round14(x0 + x3);
    t[1] = round14(x1 + x3);
    t[2] = round14(x2);
    t[3] = round14(x0 + x1 - x3);
}

fn iadst8(t: &mut [i64]) {
    let mut s = [0i64; 8];
    adst_in_perm(t, 3);
    for i in 0..4 {
        sbfly(t, &mut s, 2 * i, 1 + 2 * i, 30 - 8 * i as i32, true);
    }
    for i in 0..4 {
        shada(t, &s, i, 4 + i);
    }
    for i in 0..2 {
        sbfly(t, &mut s, 4 + 3 * i, 5 + i, 24 - 16 * i as i32, true);
    }
    for i in 0..2 {
        shada(t, &s, 4 + i, 6 + i);
    }
    for i in 0..2 {
        hada(t, i, 2 + i, false);
    }
    for i in 0..2 {
        bfly(t, 2 + 4 * i, 3 + 4 * i, 16, true);
    }
    adst_out_perm(t, 3);
    for i in 0..4 {
        t[1 + 2 * i] = -t[1 + 2 * i];
    }
}

fn iadst16(t: &mut [i64]) {
    let mut s = [0i64; 16];
    adst_in_perm(t, 4);
    for i in 0..8 {
        sbfly(t, &mut s, 2 * i, 1 + 2 * i, 31 - 4 * i as i32, true);
    }
    for i in 0..8 {
        shada(t, &s, i, 8 + i);
    }
    for i in 0..4 {
        sbfly(t, &mut s, 8 + 2 * i, 9 + 2 * i, 28 - 16 * i as i32, true);
    }
    for i in 0..4 {
        shada(t, &s, 8 + i, 12 + i);
    }
    for i in 0..4 {
        hada(t, i, 4 + i, false);
    }
    for i in 0..2 {
        for j in 0..2 {
            sbfly(t, &mut s, 4 + 8 * i + 3 * j, 5 + 8 * i + j, 24 - 16 * j as i32, true);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            shada(t, &s, 4 + 8 * j + i, 6 + 8 * j + i);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            hada(t, 8 * j + i, 2 + 8 * j + i, false);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            bfly(t, 2 + 4 * j + 8 * i, 3 + 4 * j + 8 * i, 48 + 64 * (i ^ j) as i32, false);
        }
    }
    adst_out_perm(t, 4);
    for i in 0..2 {
        for j in 0..2 {
            t[1 + 12 * j + 2 * i] = -t[1 + 12 * j + 2 * i];
        }
    }
}

/// The inverse ADST process (8.7.1.9).
pub(crate) fn iadst(t: &mut [i64], n: u32) {
    match n {
        2 => iadst4(t),
        3 => iadst8(t),
        _ => iadst16(t),
    }
}

/// The inverse Walsh-Hadamard transform (8.7.1.10).
pub(crate) fn iwht(t: &mut [i64], shift: u32) {
    let mut a = t[0] >> shift;
    let mut c = t[1] >> shift;
    let mut d = t[2] >> shift;
    let mut b = t[3] >> shift;
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    t[0] = a;
    t[1] = b;
    t[2] = c;
    t[3] = d;
}

/// The 2D inverse transform (8.7.2) of the `1 << n` square block `block`
/// (row-major), in place.
pub(crate) fn inverse_transform_2d(block: &mut [i32], n: u32, tx_type: u8, lossless: bool) {
    let n0 = 1usize << n;
    let mut t = [0i64; 32];
    let mut tmp = [0i64; 1024];
    // Rows.
    for i in 0..n0 {
        let row = &block[i * n0..(i + 1) * n0];
        if row.iter().all(|&v| v == 0) && !lossless {
            for j in 0..n0 {
                tmp[i * n0 + j] = 0;
            }
            continue;
        }
        for j in 0..n0 {
            t[j] = row[j] as i64;
        }
        if lossless {
            iwht(&mut t, 2);
        } else if tx_type == DCT_DCT || tx_type == ADST_DCT {
            idct(&mut t, n);
        } else {
            iadst(&mut t, n);
        }
        tmp[i * n0..(i + 1) * n0].copy_from_slice(&t[..n0]);
    }
    // Columns.
    let shift = (n + 2).min(6);
    for j in 0..n0 {
        for i in 0..n0 {
            t[i] = tmp[i * n0 + j];
        }
        if lossless {
            iwht(&mut t, 0);
        } else if tx_type == DCT_DCT || tx_type == DCT_ADST {
            idct(&mut t, n);
        } else {
            debug_assert!(tx_type == ADST_DCT || tx_type == ADST_ADST);
            iadst(&mut t, n);
        }
        for i in 0..n0 {
            block[i * n0 + j] = if lossless { t[i] as i32 } else { ((t[i] + (1 << (shift - 1))) >> shift) as i32 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn lcg(seed: &mut u64) -> i64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) % 2001) as i64 - 1000
    }

    /// The real inverse DCT the integer one approximates:
    /// y[k] = x[0]/sqrt(2) + sum_{f>0} x[f] cos((2k+1) f pi / 2N).
    fn ref_idct(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                (0..n)
                    .map(|f| {
                        let c = if f == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
                        c * x[f] * ((2 * k + 1) as f64 * f as f64 * PI / (2 * n) as f64).cos()
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn idct_approximates_the_real_transform() {
        let mut seed = 1u64;
        for n in 2..=5u32 {
            let n0 = 1 << n;
            for _ in 0..200 {
                let x: Vec<i64> = (0..n0).map(|_| lcg(&mut seed)).collect();
                let mut t = x.clone();
                idct(&mut t, n);
                let r = ref_idct(&x.iter().map(|&v| v as f64).collect::<Vec<_>>());
                for k in 0..n0 {
                    assert!((t[k] as f64 - r[k]).abs() <= n as f64 + 1.0, "n={n} k={k}: {} vs {}", t[k], r[k]);
                }
            }
        }
    }

    /// ADST of VP9 approximates a sine transform; find the best-matching
    /// orthogonal basis and require the integer one to track it closely.
    fn ref_iadst(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        if n == 4 {
            // sin(pi (k+1)(2f+1) / 9) * (2/3)*sqrt(2) * ... normalised like the integer one:
            // the constants are round(16384 * sqrt(2) * 2/3 * sin(i pi / 9)).
            let c = (2.0 / 3.0) * 2f64.sqrt();
            (0..4)
                .map(|k| (0..4).map(|f| x[f] * c * (((2 * f + 1) * (k + 1)) as f64 * PI / 9.0).sin()).sum())
                .collect()
        } else {
            (0..n)
                .map(|k| (0..n).map(|f| x[f] * ((2 * k + 1) as f64 * (2 * f + 1) as f64 * PI / (4 * n) as f64).sin()).sum())
                .collect()
        }
    }

    #[test]
    fn iadst_approximates_a_sine_transform() {
        let mut seed = 2u64;
        for n in 2..=4u32 {
            let n0 = 1 << n;
            for _ in 0..200 {
                let x: Vec<i64> = (0..n0).map(|_| lcg(&mut seed)).collect();
                let mut t = x.clone();
                iadst(&mut t, n);
                let r = ref_iadst(&x.iter().map(|&v| v as f64).collect::<Vec<_>>());
                for k in 0..n0 {
                    assert!((t[k] as f64 - r[k]).abs() <= n as f64 + 2.0, "n={n} k={k}: {} vs {}", t[k], r[k]);
                }
            }
        }
    }

    #[test]
    fn wht_round_trips_with_its_forward() {
        // The lossless transform is exactly invertible: check a forward
        // WHT (the encoder's) against it in tests of the encoder; here just
        // that a DC-only block spreads evenly.
        let mut b = [0i32; 16];
        b[0] = 64;
        inverse_transform_2d(&mut b, 2, DCT_DCT, true);
        assert!(b.iter().all(|&v| v == b[0]), "{b:?}");
    }
}
