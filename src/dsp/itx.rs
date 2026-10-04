//! Inverse transforms (8.7): the DCT, ADST and Walsh-Hadamard butterflies
//! exactly as the specification builds them, and the 2D transform.
//!
//! The butterflies are written once, generic over [`Lane`]: a lane type is
//! one value (a scalar) or a SIMD vector of several, one per row or column
//! transformed at once. Two arithmetics are used:
//!
//! - **8 bits**: 32-bit integers with wrapping arithmetic. A conforming
//!   8-bit stream keeps every value stored in `T` within 16 bits (8.7.2),
//!   so every product of a 14-bit constant and every sum that is rounded
//!   fits 32 bits, and wrapping arithmetic is the specification's (a
//!   wrapped intermediate that is not shifted is exact modulo 2^32). The
//!   scalar and SIMD versions compute the same wrapping values for any
//!   input, conforming or not.
//! - **10 and 12 bits**: 64-bit integers, scalar. There `T` has up to 20
//!   bits and the products before rounding need more than 32.

// Loops index arrays the way the specification's formulas do.
#![allow(clippy::needless_range_loop)]

use super::Level;
use crate::consts::{ADST_DCT, DCT_ADST, DCT_DCT};

/// `cos64_lookup` (8.7.1.1), a constant so that the butterflies' angles
/// fold away (not among tools/gen_tables.py's statics for that reason).
const COS64: [i32; 33] = [
    16384, 16364, 16305, 16207, 16069, 15893, 15679, 15426, 15137, 14811, 14449, 14053, 13623,
    13160, 12665, 12140, 11585, 11003, 10394, 9760, 9102, 8423, 7723, 7005, 6270, 5520, 4756, 3981,
    3196, 2404, 1606, 804, 0,
];

const SINPI_1_9: i32 = 5283;
const SINPI_2_9: i32 = 9929;
const SINPI_3_9: i32 = 13377;
const SINPI_4_9: i32 = 15212;

/// One value, or one value of each of [`Lane::W`] transforms done at once.
///
/// # Safety
///
/// The SIMD implementations use instructions their type's instruction set
/// provides; their methods may only run on a machine that has it (the
/// dispatchers check [`super::level`]).
pub(crate) trait Lane: Copy {
    /// Values per vector: how many rows or columns are transformed at once.
    const W: usize;
    fn zero() -> Self;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, c: i32) -> Self;
    /// `(x + (1 << (n - 1))) >> n`, `n` > 0.
    fn round_shift(self, n: u32) -> Self;
    /// `x >> n`.
    fn sar(self, n: u32) -> Self;
    /// Loads `W` coefficients.
    fn load(src: &[i32]) -> Self;
    /// Transposes the `W` x `W` matrix whose rows are `v[0..W]`.
    fn transpose(v: &mut [Self]);
    /// Whether every coefficient of `W` rows of `n0` from `src` is zero.
    fn rows_zero(src: &[i32], n0: usize) -> bool {
        src[..Self::W * n0].iter().all(|&v| v == 0)
    }
    /// Adds the `W` residuals to `W` samples of `dst`, clamped to
    /// `0..=max`. The residual is clamped to +-(1 << 16) first, which
    /// changes nothing: a sample is at most 4095.
    fn add_to(self, dst: &mut [u16], max: i32);
    /// [`inverse_2d_add`] for a `1 << n` block, with the array sizes of
    /// `n` and `W` ([`lane_sizes`]).
    #[allow(clippy::too_many_arguments)]
    fn inverse_add(
        coefs: &[i32],
        n: u32,
        tx_type: u8,
        lossless: bool,
        dst: &mut [u16],
        stride: usize,
        max: i32,
    );
}

/// Implements [`Lane::inverse_add`] for a lane type of `$w` lanes: each
/// transform size gets arrays of exactly its size (`N0` values per 1D
/// transform, `N0 / W` groups of columns), nothing larger to clear.
macro_rules! lane_sizes {
    ($w:expr) => {
        #[inline(always)]
        fn inverse_add(
            coefs: &[i32],
            n: u32,
            tx_type: u8,
            lossless: bool,
            dst: &mut [u16],
            stride: usize,
            max: i32,
        ) {
            use $crate::dsp::itx::inverse_2d_add;
            match n {
                2 if $w <= 4 => inverse_2d_add::<Self, 4, { 4 / $w }>(
                    coefs, 2, tx_type, lossless, dst, stride, max,
                ),
                3 => inverse_2d_add::<Self, 8, { 8 / $w }>(
                    coefs, 3, tx_type, lossless, dst, stride, max,
                ),
                4 => inverse_2d_add::<Self, 16, { 16 / $w }>(
                    coefs, 4, tx_type, lossless, dst, stride, max,
                ),
                5 => inverse_2d_add::<Self, 32, { 32 / $w }>(
                    coefs, 5, tx_type, lossless, dst, stride, max,
                ),
                _ => unreachable!("no {}-point transform with {} lanes", 1 << n, $w),
            }
        }
    };
}
pub(crate) use lane_sizes;

#[inline(always)]
fn neg<V: Lane>(v: V) -> V {
    V::zero().sub(v)
}

#[inline(always)]
fn round14<V: Lane>(v: V) -> V {
    v.round_shift(14)
}

#[inline(always)]
const fn brev(num_bits: u32, x: usize) -> usize {
    let mut t = 0;
    let mut i = 0;
    while i < num_bits {
        let bit = (x >> i) & 1;
        t += bit << (num_bits - 1 - i);
        i += 1;
    }
    t
}

#[inline(always)]
const fn cos64(angle: i32) -> i32 {
    let a2 = angle & 127;
    match a2 {
        0..=32 => COS64[a2 as usize],
        33..=64 => -COS64[(64 - a2) as usize],
        65..=96 => -COS64[(a2 - 64) as usize],
        _ => COS64[(128 - a2) as usize],
    }
}

#[inline(always)]
const fn sin64(angle: i32) -> i32 {
    cos64(angle - 32)
}

/// B( a, b, angle, flip ).
#[inline(always)]
fn bfly<V: Lane, const N: usize>(t: &mut [V; N], a: usize, b: usize, angle: i32, flip: bool) {
    let (c, s) = (cos64(angle), sin64(angle));
    let x = round14(t[a].mul(c).sub(t[b].mul(s)));
    let y = round14(t[a].mul(s).add(t[b].mul(c)));
    if flip {
        t[a] = y;
        t[b] = x;
    } else {
        t[a] = x;
        t[b] = y;
    }
}

/// H( a, b, flip ).
#[inline(always)]
fn hada<V: Lane, const N: usize>(t: &mut [V; N], a: usize, b: usize, flip: bool) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let x = t[a];
    let y = t[b];
    t[a] = x.add(y);
    t[b] = x.sub(y);
}

/// SB( a, b, angle, flip ).
#[inline(always)]
fn sbfly<V: Lane, const N: usize>(
    t: &[V; N],
    s: &mut [V; N],
    a: usize,
    b: usize,
    angle: i32,
    flip: bool,
) {
    let (c, sn) = (cos64(angle), sin64(angle));
    let x = t[a].mul(c).sub(t[b].mul(sn));
    let y = t[a].mul(sn).add(t[b].mul(c));
    if flip {
        s[a] = y;
        s[b] = x;
    } else {
        s[a] = x;
        s[b] = y;
    }
}

/// SH( a, b ).
#[inline(always)]
fn shada<V: Lane, const N: usize>(t: &mut [V; N], s: &[V; N], a: usize, b: usize) {
    t[a] = round14(s[a].add(s[b]));
    t[b] = round14(s[a].sub(s[b]));
}

/// The steps of the inverse DCT process (8.7.1.3) after the recursive call,
/// for a constant `n`.
#[inline(always)]
fn idct_stage<V: Lane, const N: usize>(t: &mut [V; N], n: u32) {
    let n0 = 1usize << n;
    let n1 = n0 >> 1;
    let n2 = n0 >> 2;
    let n3 = n0 >> 3;
    let nn = n as usize;
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
                bfly(
                    t,
                    n0 - nn + 3 - n2 * j - 4 * i,
                    n1 + nn - 4 + n2 * j + 4 * i,
                    28 - 16 * i as i32 + 56 * j as i32,
                    true,
                );
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
                bfly(
                    t,
                    n0 - nn + 2 - i - n2 * j,
                    n1 + nn - 3 + i + n2 * j,
                    24 + 48 * j as i32,
                    true,
                );
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

#[inline(always)]
fn idct4_core<V: Lane, const N: usize>(t: &mut [V; N]) {
    bfly(t, 0, 1, 16, true);
    idct_stage(t, 2);
}

#[inline(always)]
fn idct8_core<V: Lane, const N: usize>(t: &mut [V; N]) {
    idct4_core(t);
    idct_stage(t, 3);
}

#[inline(always)]
fn idct16_core<V: Lane, const N: usize>(t: &mut [V; N]) {
    idct8_core(t);
    idct_stage(t, 4);
}

#[inline(always)]
fn idct32_core<V: Lane, const N: usize>(t: &mut [V; N]) {
    idct16_core(t);
    idct_stage(t, 5);
}

/// Inverse DCT array permutation (8.7.1.2) then the inverse DCT, for a
/// constant `n`.
#[inline(always)]
fn idct<V: Lane, const N: usize>(t: &mut [V; N], n: u32) {
    let n0 = 1usize << n;
    let copy = *t;
    for i in 0..n0 {
        t[i] = copy[brev(n, i)];
    }
    match n {
        2 => idct4_core(t),
        3 => idct8_core(t),
        4 => idct16_core(t),
        _ => idct32_core(t),
    }
}

#[inline(always)]
fn adst_in_perm<V: Lane, const N: usize>(t: &mut [V; N], n: u32) {
    let n0 = 1usize << n;
    let n1 = n0 >> 1;
    let copy = *t;
    for i in 0..n1 {
        t[2 * i] = copy[n0 - 1 - 2 * i];
        t[2 * i + 1] = copy[2 * i];
    }
}

#[inline(always)]
fn adst_out_perm<V: Lane, const N: usize>(t: &mut [V; N], n: u32) {
    let copy = *t;
    if n == 4 {
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    for d in 0..2 {
                        t[8 * a + 4 * b + 2 * c + d] =
                            copy[8 * (d ^ c) + 4 * (c ^ b) + 2 * (b ^ a) + a];
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

#[inline(always)]
fn iadst4<V: Lane, const N: usize>(t: &mut [V; N]) {
    let s0 = t[0].mul(SINPI_1_9);
    let s1 = t[0].mul(SINPI_2_9);
    let s2 = t[1].mul(SINPI_3_9);
    let s3 = t[2].mul(SINPI_4_9);
    let s4 = t[2].mul(SINPI_1_9);
    let s5 = t[3].mul(SINPI_2_9);
    let s6 = t[3].mul(SINPI_4_9);
    let v = t[0].sub(t[2]).add(t[3]);
    let s7 = v.mul(SINPI_3_9);
    let x0 = s0.add(s3).add(s5);
    let x1 = s1.sub(s4).sub(s6);
    let x2 = s7;
    let x3 = s2;
    t[0] = round14(x0.add(x3));
    t[1] = round14(x1.add(x3));
    t[2] = round14(x2);
    t[3] = round14(x0.add(x1).sub(x3));
}

#[inline(always)]
fn iadst8<V: Lane, const N: usize>(t: &mut [V; N]) {
    let mut s = [V::zero(); N];
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
        t[1 + 2 * i] = neg(t[1 + 2 * i]);
    }
}

#[inline(always)]
fn iadst16<V: Lane, const N: usize>(t: &mut [V; N]) {
    let mut s = [V::zero(); N];
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
            sbfly(
                t,
                &mut s,
                4 + 8 * i + 3 * j,
                5 + 8 * i + j,
                24 - 16 * j as i32,
                true,
            );
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
            bfly(
                t,
                2 + 4 * j + 8 * i,
                3 + 4 * j + 8 * i,
                48 + 64 * (i ^ j) as i32,
                false,
            );
        }
    }
    adst_out_perm(t, 4);
    for i in 0..2 {
        for j in 0..2 {
            t[1 + 12 * j + 2 * i] = neg(t[1 + 12 * j + 2 * i]);
        }
    }
}

/// The inverse ADST process (8.7.1.9), for a constant `n`.
#[inline(always)]
fn iadst<V: Lane, const N: usize>(t: &mut [V; N], n: u32) {
    match n {
        2 => iadst4(t),
        3 => iadst8(t),
        _ => iadst16(t),
    }
}

/// The inverse Walsh-Hadamard transform (8.7.1.10).
#[inline(always)]
fn iwht<V: Lane, const N: usize>(t: &mut [V; N], shift: u32) {
    let mut a = t[0].sar(shift);
    let mut c = t[1].sar(shift);
    let mut d = t[2].sar(shift);
    let mut b = t[3].sar(shift);
    a = a.add(c);
    d = d.sub(b);
    let e = a.sub(d).sar(1);
    b = e.sub(b);
    c = e.sub(c);
    a = a.sub(b);
    d = d.add(c);
    t[0] = a;
    t[1] = b;
    t[2] = c;
    t[3] = d;
}

/// The 1D transform of a pass, for a constant `n`.
#[inline(always)]
fn tx1d<V: Lane, const N: usize>(
    t: &mut [V; N],
    n: u32,
    dct: bool,
    lossless: bool,
    wht_shift: u32,
) {
    if lossless {
        iwht(t, wht_shift);
    } else if dct {
        idct(t, n);
    } else {
        iadst(t, n);
    }
}

/// The 2D inverse transform (8.7.2) of the `1 << n` square block `coefs`
/// (row-major), added to the samples at `dst` (row stride `stride`) and
/// clamped to `bit_depth` bits; `n` is constant where this is inlined.
///
/// Rows of `Lane::W` are transformed together: their coefficients are
/// loaded and transposed so that a vector holds one column position of each
/// row; the results are transposed back so that a vector holds `W` columns
/// of one row, which is what the column transforms take.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn inverse_2d_add<V: Lane, const N0: usize, const G: usize>(
    coefs: &[i32],
    n: u32,
    tx_type: u8,
    lossless: bool,
    dst: &mut [u16],
    stride: usize,
    max: i32,
) {
    let n0 = 1usize << n;
    let w = V::W;
    debug_assert!(n0 == N0 && G * w == N0);
    // cols[g][i]: row i of the row transforms' output, columns g*W.. .
    let mut cols = [[V::zero(); N0]; G];
    let row_dct = tx_type == DCT_DCT || tx_type == ADST_DCT;
    let col_dct = tx_type == DCT_DCT || tx_type == DCT_ADST;
    let mut t = [V::zero(); N0];
    let mut tile = [V::zero(); 8];
    for rg in 0..G {
        let r0 = rg * w;
        if !lossless && V::rows_zero(&coefs[r0 * n0..], n0) {
            // Zero rows transform to zero; `cols` is zero already.
            continue;
        }
        for kg in 0..G {
            for l in 0..w {
                tile[l] = V::load(&coefs[(r0 + l) * n0 + kg * w..]);
            }
            V::transpose(&mut tile[..w]);
            t[kg * w..kg * w + w].copy_from_slice(&tile[..w]);
        }
        tx1d(&mut t, n, row_dct, lossless, 2);
        for kg in 0..G {
            tile[..w].copy_from_slice(&t[kg * w..kg * w + w]);
            V::transpose(&mut tile[..w]);
            cols[kg][r0..r0 + w].copy_from_slice(&tile[..w]);
        }
    }
    let shift = (n + 2).min(6);
    for (cg, col) in cols.iter().enumerate() {
        let mut t = *col;
        tx1d(&mut t, n, col_dct, lossless, 0);
        for i in 0..n0 {
            let r = if lossless {
                t[i]
            } else {
                t[i].round_shift(shift)
            };
            r.add_to(&mut dst[i * stride + cg * w..], max);
        }
    }
}

/// One 32-bit value with wrapping arithmetic: the 8-bit scalar lane.
#[derive(Clone, Copy)]
pub(crate) struct S32(i32);

impl Lane for S32 {
    const W: usize = 1;
    #[inline(always)]
    fn zero() -> Self {
        S32(0)
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        S32(self.0.wrapping_add(o.0))
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        S32(self.0.wrapping_sub(o.0))
    }
    #[inline(always)]
    fn mul(self, c: i32) -> Self {
        S32(self.0.wrapping_mul(c))
    }
    #[inline(always)]
    fn round_shift(self, n: u32) -> Self {
        S32(self.0.wrapping_add(1 << (n - 1)) >> n)
    }
    #[inline(always)]
    fn sar(self, n: u32) -> Self {
        S32(self.0 >> n)
    }
    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        S32(src[0])
    }
    #[inline(always)]
    fn transpose(_: &mut [Self]) {}
    #[inline(always)]
    fn add_to(self, dst: &mut [u16], max: i32) {
        let r = self.0.clamp(-(1 << 16), 1 << 16);
        dst[0] = (dst[0] as i32 + r).clamp(0, max) as u16;
    }
    lane_sizes!(1);
}

/// One 64-bit value: the lane above 8 bits, the specification's arithmetic
/// on a conforming stream's values (whose products need more than 32 bits).
#[derive(Clone, Copy)]
pub(crate) struct S64(i64);

impl Lane for S64 {
    const W: usize = 1;
    #[inline(always)]
    fn zero() -> Self {
        S64(0)
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        S64(self.0.wrapping_add(o.0))
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        S64(self.0.wrapping_sub(o.0))
    }
    #[inline(always)]
    fn mul(self, c: i32) -> Self {
        S64(self.0.wrapping_mul(c as i64))
    }
    #[inline(always)]
    fn round_shift(self, n: u32) -> Self {
        S64(self.0.wrapping_add(1 << (n - 1)) >> n)
    }
    #[inline(always)]
    fn sar(self, n: u32) -> Self {
        S64(self.0 >> n)
    }
    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        S64(src[0] as i64)
    }
    #[inline(always)]
    fn transpose(_: &mut [Self]) {}
    #[inline(always)]
    fn add_to(self, dst: &mut [u16], max: i32) {
        let r = self.0.clamp(-(1 << 16), 1 << 16) as i32;
        dst[0] = (dst[0] as i32 + r).clamp(0, max) as u16;
    }
    lane_sizes!(1);
}

fn scalar32(coefs: &[i32], n: u32, tx_type: u8, lossless: bool, dst: &mut [u16], stride: usize) {
    S32::inverse_add(coefs, n, tx_type, lossless, dst, stride, 255);
}

fn scalar64(
    coefs: &[i32],
    n: u32,
    tx_type: u8,
    lossless: bool,
    dst: &mut [u16],
    stride: usize,
    max: i32,
) {
    S64::inverse_add(coefs, n, tx_type, lossless, dst, stride, max);
}

/// The value every sample of a DCT_DCT block with only a DC coefficient
/// `dc` gets added (8 bits, 32-bit arithmetic): the row transform of
/// `[dc, 0, ...]` is constant, and so is the column transform of each
/// column `[v, 0, ...]`. The tests check this against the full transform.
#[inline]
fn dc_only_value(dc: i32, n: u32) -> i32 {
    let c = cos64(16);
    let v = S32(dc).mul(c).round_shift(14);
    let w = v.mul(c).round_shift(14);
    w.round_shift((n + 2).min(6)).0
}

/// The 2D inverse transform of the coefficients `coefs` (`1 << n` square,
/// row-major, dequantised) added to the prediction in `dst` (row stride
/// `stride`), clamped to `bit_depth` bits: the reconstruction of 8.6.2.
/// `eob` is the end of block: with 1, only the DC coefficient is nonzero.
#[allow(clippy::too_many_arguments)]
pub(crate) fn inverse_transform_add(
    level: Level,
    coefs: &[i32],
    n: u32,
    tx_type: u8,
    lossless: bool,
    eob: usize,
    bit_depth: u32,
    dst: &mut [u16],
    stride: usize,
) {
    let n0 = 1usize << n;
    debug_assert!(coefs.len() >= n0 * n0);
    debug_assert!(dst.len() >= (n0 - 1) * stride + n0);
    if bit_depth > 8 {
        scalar64(
            coefs,
            n,
            tx_type,
            lossless,
            dst,
            stride,
            (1 << bit_depth) - 1,
        );
        return;
    }
    if eob == 1 && tx_type == DCT_DCT && !lossless {
        let v = dc_only_value(coefs[0], n);
        super::pixel::add_const(level, dst, stride, n0, v);
        return;
    }
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 if n > 2 => {
            // SAFETY: the level says the machine has AVX2.
            unsafe { super::x86::itx_avx2(coefs, n, tx_type, lossless, dst, stride) }
        }
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Sse41 => {
            // SAFETY: the level says the machine has SSE4.1.
            unsafe { super::x86::itx_sse41(coefs, n, tx_type, lossless, dst, stride) }
        }
        #[cfg(target_arch = "aarch64")]
        Level::Neon => super::neon::itx(coefs, n, tx_type, lossless, dst, stride),
        _ => scalar32(coefs, n, tx_type, lossless, dst, stride),
    }
}

/// The 2D inverse transform of the `1 << n` square block `block`
/// (row-major), in place, with the specification's arithmetic: what the
/// encoder's tests invert its forward transforms with.
#[cfg(test)]
pub(crate) fn inverse_transform_2d(block: &mut [i32], n: u32, tx_type: u8, lossless: bool) {
    let n0 = 1usize << n;
    // Through the sample-adding kernel: residual + 2^16 into a zero
    // "prediction" of 2^17 - 1 would clamp; instead add to a mid value and
    // subtract it again, with a range wide enough for the tests' residuals.
    let mid = 1 << 15;
    let mut dst = vec![mid as u16; n0 * n0];
    scalar64(block, n, tx_type, lossless, &mut dst, n0, u16::MAX as i32);
    for (b, d) in block.iter_mut().zip(&dst) {
        *b = *d as i32 - mid;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{Rng, test_levels};
    use std::f64::consts::PI;

    /// The specification's process as first written in this crate: 64-bit
    /// integers, the recursion and index arithmetic of 8.7.1 as given.
    mod spec {
        use super::super::{COS64, SINPI_1_9, SINPI_2_9, SINPI_3_9, SINPI_4_9, brev};
        use crate::consts::*;

        fn cos64(angle: i32) -> i64 {
            super::super::cos64(angle) as i64
        }
        fn sin64(angle: i32) -> i64 {
            cos64(angle - 32)
        }
        fn round14(x: i64) -> i64 {
            (x + (1 << 13)) >> 14
        }
        fn bfly(t: &mut [i64], a: usize, b: usize, angle: i32, flip: bool) {
            let x = t[a] * cos64(angle) - t[b] * sin64(angle);
            let y = t[a] * sin64(angle) + t[b] * cos64(angle);
            t[a] = round14(x);
            t[b] = round14(y);
            if flip {
                t.swap(a, b);
            }
        }
        fn hada(t: &mut [i64], a: usize, b: usize, flip: bool) {
            let (a, b) = if flip { (b, a) } else { (a, b) };
            let x = t[a];
            let y = t[b];
            t[a] = x + y;
            t[b] = x - y;
        }
        fn sbfly(t: &[i64], s: &mut [i64], a: usize, b: usize, angle: i32, flip: bool) {
            s[a] = t[a] * cos64(angle) - t[b] * sin64(angle);
            s[b] = t[a] * sin64(angle) + t[b] * cos64(angle);
            if flip {
                s.swap(a, b);
            }
        }
        fn shada(t: &mut [i64], s: &[i64], a: usize, b: usize) {
            t[a] = round14(s[a] + s[b]);
            t[b] = round14(s[a] - s[b]);
        }
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
                        bfly(
                            t,
                            n0 - nn + 3 - n2 * j - 4 * i,
                            n1 + nn - 4 + n2 * j + 4 * i,
                            28 - 16 * i as i32 + 56 * j as i32,
                            true,
                        );
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
                        bfly(
                            t,
                            n0 - nn + 2 - i - n2 * j,
                            n1 + nn - 3 + i + n2 * j,
                            24 + 48 * j as i32,
                            true,
                        );
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
        pub fn idct(t: &mut [i64], n: u32) {
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
                                t[8 * a + 4 * b + 2 * c + d] =
                                    copy[8 * (d ^ c) + 4 * (c ^ b) + 2 * (b ^ a) + a];
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
        fn iadst4(t: &mut [i64]) {
            let (s1_9, s2_9, s3_9, s4_9) = (
                SINPI_1_9 as i64,
                SINPI_2_9 as i64,
                SINPI_3_9 as i64,
                SINPI_4_9 as i64,
            );
            let s0 = s1_9 * t[0];
            let s1 = s2_9 * t[0];
            let s2 = s3_9 * t[1];
            let s3 = s4_9 * t[2];
            let s4 = s1_9 * t[2];
            let s5 = s2_9 * t[3];
            let s6 = s4_9 * t[3];
            let v = t[0] - t[2] + t[3];
            let s7 = s3_9 * v;
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
                    sbfly(
                        t,
                        &mut s,
                        4 + 8 * i + 3 * j,
                        5 + 8 * i + j,
                        24 - 16 * j as i32,
                        true,
                    );
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
                    bfly(
                        t,
                        2 + 4 * j + 8 * i,
                        3 + 4 * j + 8 * i,
                        48 + 64 * (i ^ j) as i32,
                        false,
                    );
                }
            }
            adst_out_perm(t, 4);
            for i in 0..2 {
                for j in 0..2 {
                    t[1 + 12 * j + 2 * i] = -t[1 + 12 * j + 2 * i];
                }
            }
        }
        pub fn iadst(t: &mut [i64], n: u32) {
            match n {
                2 => iadst4(t),
                3 => iadst8(t),
                _ => iadst16(t),
            }
        }
        fn iwht(t: &mut [i64], shift: u32) {
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
        /// 8.7.2, then the reconstruction's add and clamp.
        pub fn reconstruct(
            block: &[i32],
            n: u32,
            tx_type: u8,
            lossless: bool,
            pred: &mut [u16],
            max: i32,
        ) {
            let _ = COS64;
            let n0 = 1usize << n;
            let mut t = [0i64; 32];
            let mut tmp = [0i64; 1024];
            for i in 0..n0 {
                let row = &block[i * n0..(i + 1) * n0];
                if row.iter().all(|&v| v == 0) && !lossless {
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
                    iadst(&mut t, n);
                }
                for i in 0..n0 {
                    let r = if lossless {
                        t[i]
                    } else {
                        (t[i] + (1 << (shift - 1))) >> shift
                    };
                    let p = &mut pred[i * n0 + j];
                    *p = (*p as i64 + r).clamp(0, max as i64) as u16;
                }
            }
        }
    }

    fn lcg(seed: &mut u64) -> i64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
                spec::idct(&mut t, n);
                let r = ref_idct(&x.iter().map(|&v| v as f64).collect::<Vec<_>>());
                for k in 0..n0 {
                    assert!(
                        (t[k] as f64 - r[k]).abs() <= n as f64 + 1.0,
                        "n={n} k={k}: {} vs {}",
                        t[k],
                        r[k]
                    );
                }
            }
        }
    }

    /// ADST of VP9 approximates a sine transform.
    fn ref_iadst(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        if n == 4 {
            let c = (2.0 / 3.0) * 2f64.sqrt();
            (0..4)
                .map(|k| {
                    (0..4)
                        .map(|f| x[f] * c * (((2 * f + 1) * (k + 1)) as f64 * PI / 9.0).sin())
                        .sum()
                })
                .collect()
        } else {
            (0..n)
                .map(|k| {
                    (0..n)
                        .map(|f| {
                            x[f] * ((2 * k + 1) as f64 * (2 * f + 1) as f64 * PI / (4 * n) as f64)
                                .sin()
                        })
                        .sum()
                })
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
                spec::iadst(&mut t, n);
                let r = ref_iadst(&x.iter().map(|&v| v as f64).collect::<Vec<_>>());
                for k in 0..n0 {
                    assert!(
                        (t[k] as f64 - r[k]).abs() <= n as f64 + 2.0,
                        "n={n} k={k}: {} vs {}",
                        t[k],
                        r[k]
                    );
                }
            }
        }
    }

    #[test]
    fn wht_dc_spreads_evenly() {
        let mut b = [0i32; 16];
        b[0] = 64;
        inverse_transform_2d(&mut b, 2, DCT_DCT, true);
        assert!(b.iter().all(|&v| v == b[0]), "{b:?}");
    }

    /// Random coefficient blocks: a few nonzero up to a magnitude, or dense;
    /// sometimes only the first rows, sometimes only the DC.
    fn block(rng: &mut Rng, n: u32, mag: i64) -> Vec<i32> {
        let n0 = 1usize << n;
        let mut b = vec![0i32; n0 * n0];
        match rng.range(0, 4) {
            0 => b[0] = rng.range(-mag, mag) as i32,
            1 => {
                for _ in 0..rng.range(1, 6) {
                    let i = rng.range(0, (n0 * n0 - 1) as i64) as usize;
                    b[i] = rng.range(-mag, mag) as i32;
                }
            }
            2 => {
                let rows = rng.range(1, n0 as i64) as usize;
                for v in b[..rows * n0].iter_mut() {
                    *v = rng.range(-mag, mag) as i32;
                }
            }
            _ => {
                for v in b.iter_mut() {
                    *v = rng.range(-mag, mag) as i32;
                }
            }
        }
        b
    }

    fn pred(rng: &mut Rng, n0: usize, stride: usize, max: i32) -> Vec<u16> {
        (0..(n0 - 1) * stride + n0)
            .map(|_| rng.range(0, max as i64) as u16)
            .collect()
    }

    /// The bound within which a conforming stream's coefficients keep the
    /// specification's values in 32 bits at 8 bits: the 32-bit kernels
    /// equal the 64-bit process there.
    #[test]
    fn eight_bit_kernels_equal_the_specification() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut levels = vec![Level::Scalar];
        levels.extend(test_levels());
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            let types: &[u8] = if n == 5 {
                &[DCT_DCT]
            } else {
                &[DCT_DCT, ADST_DCT, DCT_ADST, crate::consts::ADST_ADST]
            };
            for &tx_type in types {
                for lossless in [false, true] {
                    if lossless && (n != 2 || tx_type != DCT_DCT) {
                        continue;
                    }
                    for iter in 0..300 {
                        // Magnitudes from small to the conformance limit of
                        // the row input (16 bits), scaled down for the larger
                        // transforms whose outputs grow.
                        let mag = [64, 1024, 4096, 32767 >> n][iter % 4];
                        let b = block(&mut rng, n, mag);
                        let eob = if b[1..].iter().all(|&v| v == 0) { 1 } else { 2 };
                        let stride = n0 + rng.range(0, 9) as usize;
                        let p = pred(&mut rng, n0, stride, 255);
                        let mut want = vec![0u16; n0 * n0];
                        for i in 0..n0 {
                            want[i * n0..i * n0 + n0]
                                .copy_from_slice(&p[i * stride..i * stride + n0]);
                        }
                        spec::reconstruct(&b, n, tx_type, lossless, &mut want, 255);
                        for &l in &levels {
                            let mut got = p.clone();
                            inverse_transform_add(
                                l, &b, n, tx_type, lossless, eob, 8, &mut got, stride,
                            );
                            for i in 0..n0 {
                                assert_eq!(
                                    &got[i * stride..i * stride + n0],
                                    &want[i * n0..i * n0 + n0],
                                    "{l:?} n={n} type={tx_type} lossless={lossless} row {i}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Any coefficients at all, conforming or not: the SIMD kernels equal
    /// the scalar 32-bit one (wrapping alike).
    #[test]
    fn eight_bit_simd_equals_scalar_on_any_input() {
        let mut rng = Rng(0x0dd_ba11_cafe);
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            for tx_type in 0..4u8 {
                if n == 5 && tx_type != DCT_DCT {
                    continue;
                }
                for iter in 0..200 {
                    let mag = [i32::MAX as i64, 1 << 20, 1 << 16, 1 << 24][iter % 4];
                    let mut b = block(&mut rng, n, mag);
                    if iter % 7 == 0 {
                        b.iter_mut().for_each(|v| *v = i32::MIN);
                    }
                    let lossless = n == 2 && tx_type == DCT_DCT && iter % 3 == 0;
                    let p = pred(&mut rng, n0, n0, 255);
                    let mut want = p.clone();
                    inverse_transform_add(
                        Level::Scalar,
                        &b,
                        n,
                        tx_type,
                        lossless,
                        2,
                        8,
                        &mut want,
                        n0,
                    );
                    for l in test_levels() {
                        let mut got = p.clone();
                        inverse_transform_add(l, &b, n, tx_type, lossless, 2, 8, &mut got, n0);
                        assert_eq!(got, want, "{l:?} n={n} type={tx_type}");
                    }
                }
            }
        }
    }

    #[test]
    fn high_bit_depth_equals_the_specification() {
        let mut rng = Rng(77);
        for bd in [10u32, 12] {
            let max = (1 << bd) - 1;
            for n in 2..=5u32 {
                let n0 = 1usize << n;
                for tx_type in 0..4u8 {
                    if n == 5 && tx_type != DCT_DCT {
                        continue;
                    }
                    for iter in 0..100 {
                        let mag = [255i64, 1 << 14, (1 << (7 + bd)) - 1][iter % 3] >> (n - 2);
                        let b = block(&mut rng, n, mag);
                        let p = pred(&mut rng, n0, n0, max);
                        let mut want = p.clone();
                        spec::reconstruct(&b, n, tx_type, false, &mut want, max);
                        let mut got = p.clone();
                        inverse_transform_add(
                            Level::Scalar,
                            &b,
                            n,
                            tx_type,
                            false,
                            2,
                            bd,
                            &mut got,
                            n0,
                        );
                        assert_eq!(got, want, "{bd}-bit n={n} type={tx_type}");
                    }
                }
            }
        }
    }

    /// The DC-only shortcut against the full transform, over every DC value
    /// of a conforming 8-bit stream (and beyond).
    #[test]
    fn dc_only_shortcut_is_exact() {
        for n in 2..=5u32 {
            let n0 = 1usize << n;
            let mut b = vec![0i32; n0 * n0];
            for dc in (-40000..=40000)
                .step_by(7)
                .chain([i32::MAX, i32::MIN, 1 << 20])
            {
                b[0] = dc;
                let mut want = vec![128u16; n0 * n0];
                S32::inverse_add(&b, n, DCT_DCT, false, &mut want, n0, 255);
                let mut got = vec![128u16; n0 * n0];
                inverse_transform_add(Level::Scalar, &b, n, DCT_DCT, false, 1, 8, &mut got, n0);
                assert_eq!(got, want, "n={n} dc={dc}");
            }
        }
    }
}
