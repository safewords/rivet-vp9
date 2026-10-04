//! x86-64 kernels: SSE4.1 and AVX2 versions of the hot scalar kernels,
//! bit-identical to them.
//!
//! Every function here is `unsafe` because it uses instructions the
//! machine may lack: callers check [`super::level`] first. Memory is
//! accessed through raw pointers only within slices whose lengths the
//! callers assert (each function says what it reads and writes).

#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;

use super::itx::{Lane, inverse_add_n};

// ---------------------------------------------------------------------
// Inverse transforms: 32-bit lanes, 8 (AVX2) or 4 (SSE4.1) at a time.

/// Eight 32-bit lanes (AVX2).
#[derive(Clone, Copy)]
pub(crate) struct V8(__m256i);

/// Four 32-bit lanes (SSE4.1).
#[derive(Clone, Copy)]
pub(crate) struct V4(__m128i);

// SAFETY (both impls): the methods are only reached from `itx_avx2` and
// `itx_sse41`, which run with the instruction sets enabled and are called
// only after the level check. Loads and stores go through slices of the
// lane count, bounds-checked.
impl Lane for V8 {
    const W: usize = 8;
    type Cols = [[V8; 32]; 4];
    #[inline(always)]
    fn cols() -> Self::Cols {
        [[V8::zero(); 32]; 4]
    }
    #[inline(always)]
    fn zero() -> Self {
        unsafe { V8(_mm256_setzero_si256()) }
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        unsafe { V8(_mm256_add_epi32(self.0, o.0)) }
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        unsafe { V8(_mm256_sub_epi32(self.0, o.0)) }
    }
    #[inline(always)]
    fn mul(self, c: i32) -> Self {
        unsafe { V8(_mm256_mullo_epi32(self.0, _mm256_set1_epi32(c))) }
    }
    #[inline(always)]
    fn round_shift(self, n: u32) -> Self {
        unsafe {
            let r = _mm256_add_epi32(self.0, _mm256_set1_epi32(1 << (n - 1)));
            V8(_mm256_sra_epi32(r, _mm_cvtsi32_si128(n as i32)))
        }
    }
    #[inline(always)]
    fn sar(self, n: u32) -> Self {
        unsafe { V8(_mm256_sra_epi32(self.0, _mm_cvtsi32_si128(n as i32))) }
    }
    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        let s = &src[..8];
        unsafe { V8(_mm256_loadu_si256(s.as_ptr() as *const __m256i)) }
    }
    #[inline(always)]
    fn transpose(v: &mut [Self]) {
        let v = &mut v[..8];
        unsafe {
            let t0 = _mm256_unpacklo_epi32(v[0].0, v[1].0);
            let t1 = _mm256_unpackhi_epi32(v[0].0, v[1].0);
            let t2 = _mm256_unpacklo_epi32(v[2].0, v[3].0);
            let t3 = _mm256_unpackhi_epi32(v[2].0, v[3].0);
            let t4 = _mm256_unpacklo_epi32(v[4].0, v[5].0);
            let t5 = _mm256_unpackhi_epi32(v[4].0, v[5].0);
            let t6 = _mm256_unpacklo_epi32(v[6].0, v[7].0);
            let t7 = _mm256_unpackhi_epi32(v[6].0, v[7].0);
            let u0 = _mm256_unpacklo_epi64(t0, t2);
            let u1 = _mm256_unpackhi_epi64(t0, t2);
            let u2 = _mm256_unpacklo_epi64(t1, t3);
            let u3 = _mm256_unpackhi_epi64(t1, t3);
            let u4 = _mm256_unpacklo_epi64(t4, t6);
            let u5 = _mm256_unpackhi_epi64(t4, t6);
            let u6 = _mm256_unpacklo_epi64(t5, t7);
            let u7 = _mm256_unpackhi_epi64(t5, t7);
            v[0] = V8(_mm256_permute2x128_si256(u0, u4, 0x20));
            v[1] = V8(_mm256_permute2x128_si256(u1, u5, 0x20));
            v[2] = V8(_mm256_permute2x128_si256(u2, u6, 0x20));
            v[3] = V8(_mm256_permute2x128_si256(u3, u7, 0x20));
            v[4] = V8(_mm256_permute2x128_si256(u0, u4, 0x31));
            v[5] = V8(_mm256_permute2x128_si256(u1, u5, 0x31));
            v[6] = V8(_mm256_permute2x128_si256(u2, u6, 0x31));
            v[7] = V8(_mm256_permute2x128_si256(u3, u7, 0x31));
        }
    }
    #[inline(always)]
    fn rows_zero(src: &[i32], n0: usize) -> bool {
        let s = &src[..8 * n0];
        unsafe {
            let mut acc = _mm256_setzero_si256();
            for c in s.as_chunks::<8>().0 {
                acc = _mm256_or_si256(acc, _mm256_loadu_si256(c.as_ptr() as *const __m256i));
            }
            _mm256_testz_si256(acc, acc) != 0
        }
    }
    #[inline(always)]
    fn add_to(self, dst: &mut [u16], max: i32) {
        let d = &mut dst[..8];
        unsafe {
            let p = _mm256_cvtepu16_epi32(_mm_loadu_si128(d.as_ptr() as *const __m128i));
            let r = _mm256_max_epi32(self.0, _mm256_set1_epi32(-(1 << 16)));
            let r = _mm256_min_epi32(r, _mm256_set1_epi32(1 << 16));
            let s = _mm256_add_epi32(p, r);
            let s = _mm256_max_epi32(s, _mm256_setzero_si256());
            let s = _mm256_min_epi32(s, _mm256_set1_epi32(max));
            let s = _mm256_permute4x64_epi64(_mm256_packus_epi32(s, s), 0b1000);
            _mm_storeu_si128(d.as_mut_ptr() as *mut __m128i, _mm256_castsi256_si128(s));
        }
    }
}

impl Lane for V4 {
    const W: usize = 4;
    type Cols = [[V4; 32]; 8];
    #[inline(always)]
    fn cols() -> Self::Cols {
        [[V4::zero(); 32]; 8]
    }
    #[inline(always)]
    fn zero() -> Self {
        unsafe { V4(_mm_setzero_si128()) }
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        unsafe { V4(_mm_add_epi32(self.0, o.0)) }
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        unsafe { V4(_mm_sub_epi32(self.0, o.0)) }
    }
    #[inline(always)]
    fn mul(self, c: i32) -> Self {
        unsafe { V4(_mm_mullo_epi32(self.0, _mm_set1_epi32(c))) }
    }
    #[inline(always)]
    fn round_shift(self, n: u32) -> Self {
        unsafe {
            let r = _mm_add_epi32(self.0, _mm_set1_epi32(1 << (n - 1)));
            V4(_mm_sra_epi32(r, _mm_cvtsi32_si128(n as i32)))
        }
    }
    #[inline(always)]
    fn sar(self, n: u32) -> Self {
        unsafe { V4(_mm_sra_epi32(self.0, _mm_cvtsi32_si128(n as i32))) }
    }
    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        let s = &src[..4];
        unsafe { V4(_mm_loadu_si128(s.as_ptr() as *const __m128i)) }
    }
    #[inline(always)]
    fn transpose(v: &mut [Self]) {
        let v = &mut v[..4];
        unsafe {
            let t0 = _mm_unpacklo_epi32(v[0].0, v[1].0);
            let t1 = _mm_unpackhi_epi32(v[0].0, v[1].0);
            let t2 = _mm_unpacklo_epi32(v[2].0, v[3].0);
            let t3 = _mm_unpackhi_epi32(v[2].0, v[3].0);
            v[0] = V4(_mm_unpacklo_epi64(t0, t2));
            v[1] = V4(_mm_unpackhi_epi64(t0, t2));
            v[2] = V4(_mm_unpacklo_epi64(t1, t3));
            v[3] = V4(_mm_unpackhi_epi64(t1, t3));
        }
    }
    #[inline(always)]
    fn rows_zero(src: &[i32], n0: usize) -> bool {
        let s = &src[..4 * n0];
        unsafe {
            let mut acc = _mm_setzero_si128();
            for c in s.as_chunks::<4>().0 {
                acc = _mm_or_si128(acc, _mm_loadu_si128(c.as_ptr() as *const __m128i));
            }
            _mm_testz_si128(acc, acc) != 0
        }
    }
    #[inline(always)]
    fn add_to(self, dst: &mut [u16], max: i32) {
        let d = &mut dst[..4];
        unsafe {
            let p = _mm_cvtepu16_epi32(_mm_loadl_epi64(d.as_ptr() as *const __m128i));
            let r = _mm_max_epi32(self.0, _mm_set1_epi32(-(1 << 16)));
            let r = _mm_min_epi32(r, _mm_set1_epi32(1 << 16));
            let s = _mm_add_epi32(p, r);
            let s = _mm_max_epi32(s, _mm_setzero_si128());
            let s = _mm_min_epi32(s, _mm_set1_epi32(max));
            _mm_storel_epi64(d.as_mut_ptr() as *mut __m128i, _mm_packus_epi32(s, s));
        }
    }
}

/// The 8-bit inverse transform and reconstruction with AVX2 (8 x 8 and
/// up).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn itx_avx2(
    coefs: &[i32],
    n: u32,
    tx_type: u8,
    lossless: bool,
    dst: &mut [u16],
    stride: usize,
) {
    inverse_add_n::<V8>(coefs, n, tx_type, lossless, dst, stride, 255);
}

/// The 8-bit inverse transform and reconstruction with SSE4.1.
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn itx_sse41(
    coefs: &[i32],
    n: u32,
    tx_type: u8,
    lossless: bool,
    dst: &mut [u16],
    stride: usize,
) {
    inverse_add_n::<V4>(coefs, n, tx_type, lossless, dst, stride, 255);
}

// ---------------------------------------------------------------------
// Inter prediction: the 8-tap passes. A pair of taps multiplies a pair of
// samples with one `pmaddwd`: samples (at most 12 bits) and taps fit 16
// bits, the sums 32.

#[inline(always)]
unsafe fn tap_pairs_128(f: &[i16; 8]) -> [__m128i; 4] {
    let p = |a: i16, b: i16| _mm_set1_epi32(((b as u16 as i32) << 16) | a as u16 as i32);
    [p(f[0], f[1]), p(f[2], f[3]), p(f[4], f[5]), p(f[6], f[7])]
}

#[inline(always)]
unsafe fn tap_pairs_256(f: &[i16; 8]) -> [__m256i; 4] {
    let p = |a: i16, b: i16| _mm256_set1_epi32(((b as u16 as i32) << 16) | a as u16 as i32);
    [p(f[0], f[1]), p(f[2], f[3]), p(f[4], f[5]), p(f[6], f[7])]
}

#[inline(always)]
unsafe fn round_clamp_128(v: __m128i, max: __m128i) -> __m128i {
    let v = _mm_srai_epi32(_mm_add_epi32(v, _mm_set1_epi32(64)), 7);
    _mm_min_epi32(_mm_max_epi32(v, _mm_setzero_si128()), max)
}

#[inline(always)]
unsafe fn round_clamp_256(v: __m256i, max: __m256i) -> __m256i {
    let v = _mm256_srai_epi32(_mm256_add_epi32(v, _mm256_set1_epi32(64)), 7);
    _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), max)
}

/// Eight outputs (or four, `narrow`) of the filter whose 8 sources for
/// output `c` are `s.add(c + t * step)`, `t` in 0..8.
#[inline(always)]
unsafe fn filter8_128(s: *const u16, step: usize, k: &[__m128i; 4], max: __m128i) -> __m128i {
    let mut lo = _mm_setzero_si128();
    let mut hi = _mm_setzero_si128();
    for (i, kk) in k.iter().enumerate() {
        let a = _mm_loadu_si128(s.add(2 * i * step) as *const __m128i);
        let b = _mm_loadu_si128(s.add((2 * i + 1) * step) as *const __m128i);
        lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), *kk));
        hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), *kk));
    }
    _mm_packus_epi32(round_clamp_128(lo, max), round_clamp_128(hi, max))
}

#[inline(always)]
unsafe fn filter4_128(s: *const u16, step: usize, k: &[__m128i; 4], max: __m128i) -> __m128i {
    let mut lo = _mm_setzero_si128();
    for (i, kk) in k.iter().enumerate() {
        let a = _mm_loadl_epi64(s.add(2 * i * step) as *const __m128i);
        let b = _mm_loadl_epi64(s.add((2 * i + 1) * step) as *const __m128i);
        lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), *kk));
    }
    let v = round_clamp_128(lo, max);
    _mm_packus_epi32(v, v)
}

/// Sixteen outputs: as [`filter8_128`] with 256-bit vectors. The in-lane
/// unpacks put outputs 0-3 and 8-11 in `lo`, 4-7 and 12-15 in `hi`; the
/// in-lane pack puts them back in order.
#[inline(always)]
unsafe fn filter16_256(s: *const u16, step: usize, k: &[__m256i; 4], max: __m256i) -> __m256i {
    let mut lo = _mm256_setzero_si256();
    let mut hi = _mm256_setzero_si256();
    for (i, kk) in k.iter().enumerate() {
        let a = _mm256_loadu_si256(s.add(2 * i * step) as *const __m256i);
        let b = _mm256_loadu_si256(s.add((2 * i + 1) * step) as *const __m256i);
        lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), *kk));
        hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), *kk));
    }
    _mm256_packus_epi32(round_clamp_256(lo, max), round_clamp_256(hi, max))
}

/// [`super::inter::h_pass_scalar`] with SSE4.1, `w` 4 or a multiple of 8.
/// Reads `src[r * src_stride + c]` for `c < w + 7`, writes `dst[r *
/// dst_stride + c]` for `c < w`, `r < rows`.
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn h_pass_sse41(
    src: &[u16],
    src_stride: usize,
    rows: usize,
    w: usize,
    f: &[i16; 8],
    max: i32,
    dst: &mut [u16],
    dst_stride: usize,
) {
    debug_assert!(w == 4 || w.is_multiple_of(8));
    let k = tap_pairs_128(f);
    let mx = _mm_set1_epi32(max);
    for r in 0..rows {
        let s = src.as_ptr().add(r * src_stride);
        let d = dst.as_mut_ptr().add(r * dst_stride);
        if w == 4 {
            _mm_storel_epi64(d as *mut __m128i, filter4_128(s, 1, &k, mx));
        } else {
            for c in (0..w).step_by(8) {
                _mm_storeu_si128(d.add(c) as *mut __m128i, filter8_128(s.add(c), 1, &k, mx));
            }
        }
    }
}

/// [`super::inter::v_pass_scalar`] with SSE4.1. Reads `src[(r + t) *
/// src_stride + c]`, `t < 8`; writes as [`h_pass_sse41`].
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn v_pass_sse41(
    src: &[u16],
    src_stride: usize,
    h: usize,
    w: usize,
    f: &[i16; 8],
    max: i32,
    dst: &mut [u16],
    dst_stride: usize,
) {
    debug_assert!(w == 4 || w.is_multiple_of(8));
    let k = tap_pairs_128(f);
    let mx = _mm_set1_epi32(max);
    for r in 0..h {
        let s = src.as_ptr().add(r * src_stride);
        let d = dst.as_mut_ptr().add(r * dst_stride);
        if w == 4 {
            _mm_storel_epi64(d as *mut __m128i, filter4_128(s, src_stride, &k, mx));
        } else {
            for c in (0..w).step_by(8) {
                let v = filter8_128(s.add(c), src_stride, &k, mx);
                _mm_storeu_si128(d.add(c) as *mut __m128i, v);
            }
        }
    }
}

/// [`h_pass_sse41`] with AVX2, `w` a multiple of 16.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn h_pass_avx2(
    src: &[u16],
    src_stride: usize,
    rows: usize,
    w: usize,
    f: &[i16; 8],
    max: i32,
    dst: &mut [u16],
    dst_stride: usize,
) {
    debug_assert!(w.is_multiple_of(16));
    let k = tap_pairs_256(f);
    let mx = _mm256_set1_epi32(max);
    for r in 0..rows {
        let s = src.as_ptr().add(r * src_stride);
        let d = dst.as_mut_ptr().add(r * dst_stride);
        for c in (0..w).step_by(16) {
            _mm256_storeu_si256(d.add(c) as *mut __m256i, filter16_256(s.add(c), 1, &k, mx));
        }
    }
}

/// [`v_pass_sse41`] with AVX2, `w` a multiple of 16.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn v_pass_avx2(
    src: &[u16],
    src_stride: usize,
    h: usize,
    w: usize,
    f: &[i16; 8],
    max: i32,
    dst: &mut [u16],
    dst_stride: usize,
) {
    debug_assert!(w.is_multiple_of(16));
    let k = tap_pairs_256(f);
    let mx = _mm256_set1_epi32(max);
    for r in 0..h {
        let s = src.as_ptr().add(r * src_stride);
        let d = dst.as_mut_ptr().add(r * dst_stride);
        for c in (0..w).step_by(16) {
            let v = filter16_256(s.add(c), src_stride, &k, mx);
            _mm256_storeu_si256(d.add(c) as *mut __m256i, v);
        }
    }
}

/// [`super::pixel::avg_scalar`] with SSE2, `w` a multiple of 8.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn avg_sse2(
    a: &[u16],
    b: &[u16],
    w: usize,
    h: usize,
    dst: &mut [u16],
    stride: usize,
) {
    assert!(w.is_multiple_of(8) && a.len() >= w * h && b.len() >= w * h);
    assert!(h == 0 || dst.len() >= (h - 1) * stride + w);
    for i in 0..h {
        let (pa, pb) = (a.as_ptr().add(i * w), b.as_ptr().add(i * w));
        let pd = dst.as_mut_ptr().add(i * stride);
        for c in (0..w).step_by(8) {
            let x = _mm_loadu_si128(pa.add(c) as *const __m128i);
            let y = _mm_loadu_si128(pb.add(c) as *const __m128i);
            _mm_storeu_si128(pd.add(c) as *mut __m128i, _mm_avg_epu16(x, y));
        }
    }
}

// ---------------------------------------------------------------------
// Loop filter: eight 16-bit lanes (SSE4.1).

/// Eight 16-bit lanes (SSE4.1).
#[derive(Clone, Copy)]
pub(crate) struct V16x8(__m128i);

// SAFETY: reached only from `lf_sse41`, which runs with SSE4.1 enabled
// after the level check; loads and stores go through 8-sample slices.
impl super::lf::V16 for V16x8 {
    #[inline(always)]
    fn splat(v: i16) -> Self {
        unsafe { V16x8(_mm_set1_epi16(v)) }
    }
    #[inline(always)]
    fn halves(a: i16, b: i16) -> Self {
        unsafe { V16x8(_mm_unpacklo_epi64(_mm_set1_epi16(a), _mm_set1_epi16(b))) }
    }
    #[inline(always)]
    fn load(src: &[u16]) -> Self {
        let s = &src[..8];
        unsafe { V16x8(_mm_loadu_si128(s.as_ptr() as *const __m128i)) }
    }
    #[inline(always)]
    fn store(self, dst: &mut [u16]) {
        let d = &mut dst[..8];
        unsafe { _mm_storeu_si128(d.as_mut_ptr() as *mut __m128i, self.0) }
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        unsafe { V16x8(_mm_add_epi16(self.0, o.0)) }
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        unsafe { V16x8(_mm_sub_epi16(self.0, o.0)) }
    }
    #[inline(always)]
    fn max(self, o: Self) -> Self {
        unsafe { V16x8(_mm_max_epi16(self.0, o.0)) }
    }
    #[inline(always)]
    fn min(self, o: Self) -> Self {
        unsafe { V16x8(_mm_min_epi16(self.0, o.0)) }
    }
    #[inline(always)]
    fn absdiff(self, o: Self) -> Self {
        unsafe {
            V16x8(_mm_or_si128(
                _mm_subs_epu16(self.0, o.0),
                _mm_subs_epu16(o.0, self.0),
            ))
        }
    }
    #[inline(always)]
    fn gt(self, o: Self) -> Self {
        unsafe { V16x8(_mm_cmpgt_epi16(self.0, o.0)) }
    }
    #[inline(always)]
    fn and(self, o: Self) -> Self {
        unsafe { V16x8(_mm_and_si128(self.0, o.0)) }
    }
    #[inline(always)]
    fn or(self, o: Self) -> Self {
        unsafe { V16x8(_mm_or_si128(self.0, o.0)) }
    }
    #[inline(always)]
    fn andnot(self, o: Self) -> Self {
        unsafe { V16x8(_mm_andnot_si128(self.0, o.0)) }
    }
    #[inline(always)]
    fn select(mask: Self, a: Self, b: Self) -> Self {
        unsafe { V16x8(_mm_blendv_epi8(b.0, a.0, mask.0)) }
    }
    #[inline(always)]
    fn sra(self, n: u32) -> Self {
        unsafe { V16x8(_mm_sra_epi16(self.0, _mm_cvtsi32_si128(n as i32))) }
    }
    #[inline(always)]
    fn srl(self, n: u32) -> Self {
        unsafe { V16x8(_mm_srl_epi16(self.0, _mm_cvtsi32_si128(n as i32))) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm_testz_si128(self.0, self.0) == 0 }
    }
    #[inline(always)]
    fn transpose(v: &mut [Self; 8]) {
        unsafe {
            let a0 = _mm_unpacklo_epi16(v[0].0, v[1].0);
            let a1 = _mm_unpackhi_epi16(v[0].0, v[1].0);
            let a2 = _mm_unpacklo_epi16(v[2].0, v[3].0);
            let a3 = _mm_unpackhi_epi16(v[2].0, v[3].0);
            let a4 = _mm_unpacklo_epi16(v[4].0, v[5].0);
            let a5 = _mm_unpackhi_epi16(v[4].0, v[5].0);
            let a6 = _mm_unpacklo_epi16(v[6].0, v[7].0);
            let a7 = _mm_unpackhi_epi16(v[6].0, v[7].0);
            let b0 = _mm_unpacklo_epi32(a0, a2);
            let b1 = _mm_unpackhi_epi32(a0, a2);
            let b2 = _mm_unpacklo_epi32(a1, a3);
            let b3 = _mm_unpackhi_epi32(a1, a3);
            let b4 = _mm_unpacklo_epi32(a4, a6);
            let b5 = _mm_unpackhi_epi32(a4, a6);
            let b6 = _mm_unpacklo_epi32(a5, a7);
            let b7 = _mm_unpackhi_epi32(a5, a7);
            v[0] = V16x8(_mm_unpacklo_epi64(b0, b4));
            v[1] = V16x8(_mm_unpackhi_epi64(b0, b4));
            v[2] = V16x8(_mm_unpacklo_epi64(b1, b5));
            v[3] = V16x8(_mm_unpackhi_epi64(b1, b5));
            v[4] = V16x8(_mm_unpacklo_epi64(b2, b6));
            v[5] = V16x8(_mm_unpackhi_epi64(b2, b6));
            v[6] = V16x8(_mm_unpacklo_epi64(b3, b7));
            v[7] = V16x8(_mm_unpackhi_epi64(b3, b7));
        }
    }
}

/// [`super::lf::edges_scalar`] with SSE4.1, eight positions at a time.
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn lf_sse41(
    buf: &mut [u16],
    stride: usize,
    x0: usize,
    y0: usize,
    vertical: bool,
    n_edges: usize,
    n_runs: usize,
    e: &super::lf::Edges,
    bit_depth: u32,
) {
    super::lf::edges_simd::<V16x8>(buf, stride, x0, y0, vertical, n_edges, n_runs, e, bit_depth);
}
