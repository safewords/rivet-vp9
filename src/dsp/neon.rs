//! aarch64 kernels: NEON versions of the hot scalar kernels, bit-identical
//! to them. NEON is part of the aarch64 baseline, so these need no run-time
//! check; `VP9_FORCE_SCALAR` still selects the scalar kernels.
//!
//! Memory is accessed through raw pointers only within slices whose
//! lengths the functions assert.

#![allow(unused_unsafe, unsafe_op_in_unsafe_fn)]

use std::arch::aarch64::*;

use super::itx::Lane;

/// Four 32-bit lanes.
#[derive(Clone, Copy)]
pub(crate) struct V4(int32x4_t);

// SAFETY: NEON is always present on aarch64; loads and stores go through
// slices of the lane count, bounds-checked.
impl Lane for V4 {
    const W: usize = 4;
    super::itx::lane_sizes!(4);
    #[inline(always)]
    fn zero() -> Self {
        unsafe { V4(vdupq_n_s32(0)) }
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        unsafe { V4(vaddq_s32(self.0, o.0)) }
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        unsafe { V4(vsubq_s32(self.0, o.0)) }
    }
    #[inline(always)]
    fn mul(self, c: i32) -> Self {
        unsafe { V4(vmulq_n_s32(self.0, c)) }
    }
    #[inline(always)]
    fn round_shift(self, n: u32) -> Self {
        unsafe {
            let r = vaddq_s32(self.0, vdupq_n_s32(1 << (n - 1)));
            V4(vshlq_s32(r, vdupq_n_s32(-(n as i32))))
        }
    }
    #[inline(always)]
    fn sar(self, n: u32) -> Self {
        unsafe { V4(vshlq_s32(self.0, vdupq_n_s32(-(n as i32)))) }
    }
    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        let s = &src[..4];
        unsafe { V4(vld1q_s32(s.as_ptr())) }
    }
    #[inline(always)]
    fn transpose(v: &mut [Self]) {
        let v = &mut v[..4];
        unsafe {
            let a = vtrnq_s32(v[0].0, v[1].0);
            let b = vtrnq_s32(v[2].0, v[3].0);
            v[0] = V4(vcombine_s32(vget_low_s32(a.0), vget_low_s32(b.0)));
            v[1] = V4(vcombine_s32(vget_low_s32(a.1), vget_low_s32(b.1)));
            v[2] = V4(vcombine_s32(vget_high_s32(a.0), vget_high_s32(b.0)));
            v[3] = V4(vcombine_s32(vget_high_s32(a.1), vget_high_s32(b.1)));
        }
    }
    #[inline(always)]
    fn add_to(self, dst: &mut [u16], max: i32) {
        let d = &mut dst[..4];
        unsafe {
            let p = vreinterpretq_s32_u32(vmovl_u16(vld1_u16(d.as_ptr())));
            let r = vmaxq_s32(self.0, vdupq_n_s32(-(1 << 16)));
            let r = vminq_s32(r, vdupq_n_s32(1 << 16));
            let s = vaddq_s32(p, r);
            let s = vminq_s32(vmaxq_s32(s, vdupq_n_s32(0)), vdupq_n_s32(max));
            vst1_u16(d.as_mut_ptr(), vqmovun_s32(s));
        }
    }
}

/// The 8-bit inverse transform and reconstruction.
pub(crate) fn itx(
    coefs: &[i32],
    n: u32,
    tx_type: u8,
    lossless: bool,
    dst: &mut [u16],
    stride: usize,
) {
    V4::inverse_add(coefs, n, tx_type, lossless, dst, stride, 255);
}

#[inline(always)]
unsafe fn round_clamp(v: int32x4_t, max: int32x4_t) -> uint16x4_t {
    let v = vshrq_n_s32::<7>(vaddq_s32(v, vdupq_n_s32(64)));
    vqmovun_s32(vminq_s32(vmaxq_s32(v, vdupq_n_s32(0)), max))
}

/// Eight outputs of the filter whose sources for output `c` are
/// `s.add(c + t * step)`.
#[inline(always)]
unsafe fn filter8(s: *const u16, step: usize, f: &[i16; 8], max: int32x4_t) -> uint16x8_t {
    let mut lo = vdupq_n_s32(0);
    let mut hi = vdupq_n_s32(0);
    for (t, &k) in f.iter().enumerate() {
        let v = vreinterpretq_s16_u16(vld1q_u16(s.add(t * step)));
        lo = vmlal_n_s16(lo, vget_low_s16(v), k);
        hi = vmlal_n_s16(hi, vget_high_s16(v), k);
    }
    vcombine_u16(round_clamp(lo, max), round_clamp(hi, max))
}

#[inline(always)]
unsafe fn filter4(s: *const u16, step: usize, f: &[i16; 8], max: int32x4_t) -> uint16x4_t {
    let mut lo = vdupq_n_s32(0);
    for (t, &k) in f.iter().enumerate() {
        let v = vreinterpret_s16_u16(vld1_u16(s.add(t * step)));
        lo = vmlal_n_s16(lo, v, k);
    }
    round_clamp(lo, max)
}

/// [`super::inter::h_pass_scalar`], `w` 4 or a multiple of 8.
#[allow(clippy::too_many_arguments)]
pub(crate) fn h_pass(
    src: &[u16],
    src_stride: usize,
    rows: usize,
    w: usize,
    f: &[i16; 8],
    max: i32,
    dst: &mut [u16],
    dst_stride: usize,
) {
    assert!(w == 4 || w.is_multiple_of(8));
    assert!(rows > 0 && src.len() >= (rows - 1) * src_stride + w + 7);
    assert!(dst.len() >= (rows - 1) * dst_stride + w);
    // SAFETY: the asserts above bound every load (row r, columns up to
    // w + 6) and store (row r, columns below w).
    unsafe {
        let mx = vdupq_n_s32(max);
        for r in 0..rows {
            let s = src.as_ptr().add(r * src_stride);
            let d = dst.as_mut_ptr().add(r * dst_stride);
            if w == 4 {
                vst1_u16(d, filter4(s, 1, f, mx));
            } else {
                for c in (0..w).step_by(8) {
                    vst1q_u16(d.add(c), filter8(s.add(c), 1, f, mx));
                }
            }
        }
    }
}

/// [`super::inter::v_pass_scalar`], `w` 4 or a multiple of 8.
#[allow(clippy::too_many_arguments)]
pub(crate) fn v_pass(
    src: &[u16],
    src_stride: usize,
    h: usize,
    w: usize,
    f: &[i16; 8],
    max: i32,
    dst: &mut [u16],
    dst_stride: usize,
) {
    assert!(w == 4 || w.is_multiple_of(8));
    assert!(h > 0 && src.len() >= (h + 6) * src_stride + w);
    assert!(dst.len() >= (h - 1) * dst_stride + w);
    // SAFETY: the asserts bound every load (rows r..r + 8) and store.
    unsafe {
        let mx = vdupq_n_s32(max);
        for r in 0..h {
            let s = src.as_ptr().add(r * src_stride);
            let d = dst.as_mut_ptr().add(r * dst_stride);
            if w == 4 {
                vst1_u16(d, filter4(s, src_stride, f, mx));
            } else {
                for c in (0..w).step_by(8) {
                    vst1q_u16(d.add(c), filter8(s.add(c), src_stride, f, mx));
                }
            }
        }
    }
}

/// [`super::pixel::avg_scalar`], `w` a multiple of 8.
pub(crate) fn avg(a: &[u16], b: &[u16], w: usize, h: usize, dst: &mut [u16], stride: usize) {
    assert!(w.is_multiple_of(8) && a.len() >= w * h && b.len() >= w * h);
    assert!(h == 0 || dst.len() >= (h - 1) * stride + w);
    // SAFETY: bounded by the asserts.
    unsafe {
        for i in 0..h {
            let (pa, pb) = (a.as_ptr().add(i * w), b.as_ptr().add(i * w));
            let pd = dst.as_mut_ptr().add(i * stride);
            for c in (0..w).step_by(8) {
                vst1q_u16(
                    pd.add(c),
                    vrhaddq_u16(vld1q_u16(pa.add(c)), vld1q_u16(pb.add(c))),
                );
            }
        }
    }
}

/// Eight 16-bit lanes, for the loop filter.
#[derive(Clone, Copy)]
pub(crate) struct V16x8(int16x8_t);

// SAFETY: NEON is always present; loads and stores go through 8-sample
// slices.
impl super::lf::V16 for V16x8 {
    const RUNS: usize = 2;
    #[inline(always)]
    fn splat(v: i16) -> Self {
        unsafe { V16x8(vdupq_n_s16(v)) }
    }
    #[inline(always)]
    fn per_run(v: [i16; 4]) -> Self {
        unsafe { V16x8(vcombine_s16(vdup_n_s16(v[0]), vdup_n_s16(v[1]))) }
    }
    #[inline(always)]
    fn load2(a: &[u16], _: &[u16]) -> Self {
        Self::load(a)
    }
    #[inline(always)]
    fn store2(self, a: &mut [u16], _: &mut [u16]) {
        self.store(a)
    }
    #[inline(always)]
    fn load(src: &[u16]) -> Self {
        let s = &src[..8];
        unsafe { V16x8(vreinterpretq_s16_u16(vld1q_u16(s.as_ptr()))) }
    }
    #[inline(always)]
    fn store(self, dst: &mut [u16]) {
        let d = &mut dst[..8];
        unsafe { vst1q_u16(d.as_mut_ptr(), vreinterpretq_u16_s16(self.0)) }
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        unsafe { V16x8(vaddq_s16(self.0, o.0)) }
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        unsafe { V16x8(vsubq_s16(self.0, o.0)) }
    }
    #[inline(always)]
    fn max(self, o: Self) -> Self {
        unsafe { V16x8(vmaxq_s16(self.0, o.0)) }
    }
    #[inline(always)]
    fn min(self, o: Self) -> Self {
        unsafe { V16x8(vminq_s16(self.0, o.0)) }
    }
    #[inline(always)]
    fn absdiff(self, o: Self) -> Self {
        unsafe {
            V16x8(vreinterpretq_s16_u16(vabdq_u16(
                vreinterpretq_u16_s16(self.0),
                vreinterpretq_u16_s16(o.0),
            )))
        }
    }
    #[inline(always)]
    fn gt(self, o: Self) -> Self {
        unsafe { V16x8(vreinterpretq_s16_u16(vcgtq_s16(self.0, o.0))) }
    }
    #[inline(always)]
    fn and(self, o: Self) -> Self {
        unsafe { V16x8(vandq_s16(self.0, o.0)) }
    }
    #[inline(always)]
    fn or(self, o: Self) -> Self {
        unsafe { V16x8(vorrq_s16(self.0, o.0)) }
    }
    #[inline(always)]
    fn andnot(self, o: Self) -> Self {
        unsafe { V16x8(vbicq_s16(o.0, self.0)) }
    }
    #[inline(always)]
    fn select(mask: Self, a: Self, b: Self) -> Self {
        unsafe { V16x8(vbslq_s16(vreinterpretq_u16_s16(mask.0), a.0, b.0)) }
    }
    #[inline(always)]
    fn sra(self, n: u32) -> Self {
        unsafe { V16x8(vshlq_s16(self.0, vdupq_n_s16(-(n as i16)))) }
    }
    #[inline(always)]
    fn srl(self, n: u32) -> Self {
        unsafe {
            V16x8(vreinterpretq_s16_u16(vshlq_u16(
                vreinterpretq_u16_s16(self.0),
                vdupq_n_s16(-(n as i16)),
            )))
        }
    }
    #[inline(always)]
    fn any(self) -> bool {
        unsafe { vmaxvq_u16(vreinterpretq_u16_s16(self.0)) != 0 }
    }
    #[inline(always)]
    fn transpose(v: &mut [Self; 8]) {
        unsafe {
            let t0 = vtrnq_s16(v[0].0, v[1].0);
            let t1 = vtrnq_s16(v[2].0, v[3].0);
            let t2 = vtrnq_s16(v[4].0, v[5].0);
            let t3 = vtrnq_s16(v[6].0, v[7].0);
            let u0 = vtrnq_s32(vreinterpretq_s32_s16(t0.0), vreinterpretq_s32_s16(t1.0));
            let u1 = vtrnq_s32(vreinterpretq_s32_s16(t0.1), vreinterpretq_s32_s16(t1.1));
            let u2 = vtrnq_s32(vreinterpretq_s32_s16(t2.0), vreinterpretq_s32_s16(t3.0));
            let u3 = vtrnq_s32(vreinterpretq_s32_s16(t2.1), vreinterpretq_s32_s16(t3.1));
            v[0] = V16x8(vreinterpretq_s16_s32(vcombine_s32(
                vget_low_s32(u0.0),
                vget_low_s32(u2.0),
            )));
            v[1] = V16x8(vreinterpretq_s16_s32(vcombine_s32(
                vget_low_s32(u1.0),
                vget_low_s32(u3.0),
            )));
            v[2] = V16x8(vreinterpretq_s16_s32(vcombine_s32(
                vget_low_s32(u0.1),
                vget_low_s32(u2.1),
            )));
            v[3] = V16x8(vreinterpretq_s16_s32(vcombine_s32(
                vget_low_s32(u1.1),
                vget_low_s32(u3.1),
            )));
            v[4] = V16x8(vreinterpretq_s16_s32(vcombine_s32(
                vget_high_s32(u0.0),
                vget_high_s32(u2.0),
            )));
            v[5] = V16x8(vreinterpretq_s16_s32(vcombine_s32(
                vget_high_s32(u1.0),
                vget_high_s32(u3.0),
            )));
            v[6] = V16x8(vreinterpretq_s16_s32(vcombine_s32(
                vget_high_s32(u0.1),
                vget_high_s32(u2.1),
            )));
            v[7] = V16x8(vreinterpretq_s16_s32(vcombine_s32(
                vget_high_s32(u1.1),
                vget_high_s32(u3.1),
            )));
        }
    }
}

/// [`crate::encoder::fdct::forward_scalar`]: the two 16-bit matrix
/// products with widening multiply-accumulates (exact, as the scalar code's
/// 32-bit sums).
pub(crate) fn fdct(t: &crate::encoder::fdct::Fwd2d, res: &[i16], out: &mut [i32]) {
    let n0 = t.fc.n0;
    assert!(res.len() >= n0 * n0 && out.len() >= n0 * n0);
    assert!(t.fc.q.len() >= n0 * n0 && t.fr.qt.len() >= n0 * n0);
    let mut u = [0i16; 32 * 32];
    // SAFETY: every load and store is within the asserted lengths (rows of
    // n0 values, in chunks of 4 or 8 that divide n0).
    unsafe {
        let rnd = vdupq_n_s32(1 << (t.shift1 - 1));
        let sh = vdupq_n_s32(-(t.shift1 as i32));
        let rp = res.as_ptr();
        for f in 0..n0 {
            let fc = &t.fc.q[f * n0..f * n0 + n0];
            for c in (0..n0).step_by(4) {
                let mut acc = rnd;
                for (k, &f) in fc.iter().enumerate() {
                    acc = vmlal_n_s16(acc, vld1_s16(rp.add(k * n0 + c)), f);
                }
                vst1_s16(
                    u.as_mut_ptr().add(f * n0 + c),
                    vqmovn_s32(vshlq_s32(acc, sh)),
                );
            }
        }
        let qt = t.fr.qt.as_ptr();
        for f in 0..n0 {
            for g in (0..n0).step_by(4) {
                let mut acc = vdupq_n_s32(0);
                for j in 0..n0 {
                    acc = vmlal_n_s16(acc, vld1_s16(qt.add(j * n0 + g)), u[f * n0 + j]);
                }
                vst1q_s32(out.as_mut_ptr().add(f * n0 + g), acc);
            }
        }
    }
}

/// [`super::pixel::sse_scalar`], `w` at least 8.
pub(crate) fn sse(a: &[u16], sa: usize, b: &[u16], sb: usize, w: usize, h: usize) -> u64 {
    assert!(a.len() >= (h - 1) * sa + w && b.len() >= (h - 1) * sb + w);
    let w8 = w & !7;
    let mut tail = 0u64;
    // SAFETY: reads within the asserted rows.
    unsafe {
        let mut total = vdupq_n_u64(0);
        for i in 0..h {
            let (pa, pb) = (a.as_ptr().add(i * sa), b.as_ptr().add(i * sb));
            for c in (0..w8).step_by(8) {
                let d = vabdq_u16(vld1q_u16(pa.add(c)), vld1q_u16(pb.add(c)));
                let lo = vmull_u16(vget_low_u16(d), vget_low_u16(d));
                let hi = vmull_u16(vget_high_u16(d), vget_high_u16(d));
                total = vpadalq_u32(total, lo);
                total = vpadalq_u32(total, hi);
            }
            for c in w8..w {
                let d = *pa.add(c) as i64 - *pb.add(c) as i64;
                tail += (d * d) as u64;
            }
        }
        vaddvq_u64(total) + tail
    }
}

/// [`super::pixel::sad_scalar`], `w` at least 8.
pub(crate) fn sad(a: &[u16], sa: usize, b: &[u16], sb: usize, w: usize, h: usize) -> u64 {
    assert!(a.len() >= (h - 1) * sa + w && b.len() >= (h - 1) * sb + w);
    let w8 = w & !7;
    let mut tail = 0u64;
    // SAFETY: reads within the asserted rows.
    unsafe {
        let mut total = vdupq_n_u64(0);
        for i in 0..h {
            let (pa, pb) = (a.as_ptr().add(i * sa), b.as_ptr().add(i * sb));
            let mut row = vdupq_n_u32(0);
            for c in (0..w8).step_by(8) {
                let d = vabdq_u16(vld1q_u16(pa.add(c)), vld1q_u16(pb.add(c)));
                row = vpadalq_u16(row, d);
            }
            total = vpadalq_u32(total, row);
            for c in w8..w {
                tail += (*pa.add(c) as i32 - *pb.add(c) as i32).unsigned_abs() as u64;
            }
        }
        vaddvq_u64(total) + tail
    }
}

/// [`crate::encoder::fdct::quantize_scalar`]: four at a time, the same
/// single-precision operations (separate multiply and add).
pub(crate) fn quantize(d: &[i32], scale: f32, bias: f32, max_coef: i32, out: &mut [i32]) {
    let len = d.len().min(out.len());
    let len4 = len & !3;
    // SAFETY: loads and stores below `len4`, within both slices.
    unsafe {
        let sc = vdupq_n_f32(scale);
        let bi = vdupq_n_f32(bias);
        let mx = vdupq_n_s32(max_coef);
        for i in (0..len4).step_by(4) {
            let v = vld1q_s32(d.as_ptr().add(i));
            let t = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vabsq_s32(v)), sc), bi);
            let l = vminq_s32(vcvtq_s32_f32(t), mx);
            let neg = vcltq_s32(v, vdupq_n_s32(0));
            vst1q_s32(out.as_mut_ptr().add(i), vbslq_s32(neg, vnegq_s32(l), l));
        }
    }
    crate::encoder::fdct::quantize_scalar(
        &d[len4..len],
        scale,
        bias,
        max_coef,
        &mut out[len4..len],
    );
}

/// Two 64-bit lanes, for the inverse transforms above 8 bits. The
/// multiplies are vmull_s32 (low 32 bits, sign-extended, times the
/// constant), the scalar lane's arithmetic (itx.rs, `S64`).
#[derive(Clone, Copy)]
pub(crate) struct V64x2(int64x2_t);

// SAFETY: NEON is always present; loads and stores go through
// bounds-checked slices.
impl Lane for V64x2 {
    const W: usize = 2;
    super::itx::lane_sizes!(2);
    #[inline(always)]
    fn zero() -> Self {
        unsafe { V64x2(vdupq_n_s64(0)) }
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        unsafe { V64x2(vaddq_s64(self.0, o.0)) }
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        unsafe { V64x2(vsubq_s64(self.0, o.0)) }
    }
    #[inline(always)]
    fn mul(self, c: i32) -> Self {
        unsafe { V64x2(vmull_n_s32(vmovn_s64(self.0), c)) }
    }
    #[inline(always)]
    fn round_shift(self, n: u32) -> Self {
        unsafe { V64x2(vaddq_s64(self.0, vdupq_n_s64(1 << (n - 1)))).sar(n) }
    }
    #[inline(always)]
    fn sar(self, n: u32) -> Self {
        unsafe { V64x2(vshlq_s64(self.0, vdupq_n_s64(-(n as i64)))) }
    }
    #[inline(always)]
    fn load(src: &[i32]) -> Self {
        let s = &src[..2];
        unsafe { V64x2(vmovl_s32(vld1_s32(s.as_ptr()))) }
    }
    #[inline(always)]
    fn transpose(v: &mut [Self]) {
        let v = &mut v[..2];
        unsafe {
            let (a, b) = (v[0].0, v[1].0);
            v[0] = V64x2(vtrn1q_s64(a, b));
            v[1] = V64x2(vtrn2q_s64(a, b));
        }
    }
    #[inline(always)]
    fn add_to(self, dst: &mut [u16], max: i32) {
        let d = &mut dst[..2];
        let mut r = [0i64; 2];
        unsafe { vst1q_s64(r.as_mut_ptr(), self.0) };
        for (d, r) in d.iter_mut().zip(r) {
            let r = r.clamp(-(1 << 16), 1 << 16) as i32;
            *d = (*d as i32 + r).clamp(0, max) as u16;
        }
    }
}

/// The inverse transform and reconstruction above 8 bits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn itx64(
    coefs: &[i32],
    n: u32,
    tx_type: u8,
    lossless: bool,
    dst: &mut [u16],
    stride: usize,
    max: i32,
) {
    V64x2::inverse_add(coefs, n, tx_type, lossless, dst, stride, max);
}

/// [`super::inter::h_scaled_scalar`]: each output's 8 samples times its
/// own taps, widened and summed across the vector. The caller bounds the
/// reads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn h_scaled(
    src: &[u16],
    x: i32,
    x_step: i32,
    px0: i32,
    filter: u8,
    max: i32,
    out: &mut [u16],
) {
    for (c, o) in out.iter_mut().enumerate() {
        let p = x + x_step * c as i32;
        let base = ((p >> 4) - 3 - px0) as usize;
        let t = super::inter::taps(filter, (p & 15) as usize);
        let s = &src[base..base + 8];
        // SAFETY: 8 samples and 8 taps, bounds-checked slices.
        let sum = unsafe {
            let v = vreinterpretq_s16_u16(vld1q_u16(s.as_ptr()));
            let k = vld1q_s16(t.as_ptr());
            let acc = vmull_s16(vget_low_s16(v), vget_low_s16(k));
            vaddvq_s32(vmlal_s16(acc, vget_high_s16(v), vget_high_s16(k)))
        };
        *o = ((sum + 64) >> 7).clamp(0, max) as u16;
    }
}
