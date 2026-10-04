//! The sample-level kernels: inverse transforms, intra and inter
//! prediction, loop filter, and the encoder's distortion and forward
//! transform kernels. The decoder above them never touches samples
//! directly.
//!
//! Every kernel has a scalar version written to the specification's
//! formulas, and the hot ones have SIMD versions (x86-64: SSE4.1 and AVX2;
//! aarch64: NEON) that give bit-identical results — the kernels' tests
//! check them against the scalar ones on random and extreme input. The
//! instruction set is picked once at run time ([`level`]);
//! `VP9_FORCE_SCALAR=1` in the environment forces the scalar kernels.

pub(crate) mod inter;
pub(crate) mod intra;
pub(crate) mod itx;
pub(crate) mod lf;
pub(crate) mod pixel;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

use std::sync::atomic::{AtomicU8, Ordering};

/// An instruction set the kernels have a version for, in increasing order
/// of preference on its architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Level {
    /// Plain Rust.
    Scalar,
    /// x86-64 with SSE4.1.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    Sse41,
    /// x86-64 with AVX2 (and SSE4.1).
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    Avx2,
    /// aarch64 with NEON (always present there).
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    Neon,
}

const UNSET: u8 = 0xff;
static LEVEL: AtomicU8 = AtomicU8::new(UNSET);

fn from_u8(v: u8) -> Level {
    match v {
        1 => Level::Sse41,
        2 => Level::Avx2,
        3 => Level::Neon,
        _ => Level::Scalar,
    }
}

/// The best instruction set this machine has, ignoring `VP9_FORCE_SCALAR`.
pub(crate) fn detected() -> Level {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("sse4.1") {
            return Level::Avx2;
        }
        if std::is_x86_feature_detected!("sse4.1") {
            return Level::Sse41;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Level::Neon;
        }
    }
    Level::Scalar
}

/// The instruction set the kernels use: [`detected`], unless
/// `VP9_FORCE_SCALAR` is set (to anything but `0` or empty).
#[inline]
pub(crate) fn level() -> Level {
    let v = LEVEL.load(Ordering::Relaxed);
    if v != UNSET {
        return from_u8(v);
    }
    init_level()
}

#[cold]
fn init_level() -> Level {
    let forced = std::env::var("VP9_FORCE_SCALAR").is_ok_and(|v| !v.is_empty() && v != "0");
    let l = if forced { Level::Scalar } else { detected() };
    LEVEL.store(l as u8, Ordering::Relaxed);
    l
}

/// The SIMD levels this machine can run, for the kernels' tests: each is
/// checked against the scalar kernel. With `VP9_REQUIRE_SIMD` set, a
/// machine without the level its architecture is expected to have (AVX2 on
/// x86-64, NEON on aarch64) fails the tests instead of skipping the SIMD
/// kernels silently.
#[cfg(test)]
pub(crate) fn test_levels() -> Vec<Level> {
    let best = detected();
    if std::env::var("VP9_REQUIRE_SIMD").is_ok_and(|v| !v.is_empty() && v != "0") {
        #[cfg(target_arch = "x86_64")]
        assert_eq!(best, Level::Avx2, "VP9_REQUIRE_SIMD: AVX2 not available");
        #[cfg(target_arch = "aarch64")]
        assert_eq!(best, Level::Neon, "VP9_REQUIRE_SIMD: NEON not available");
    }
    let mut v = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if best >= Level::Sse41 {
            v.push(Level::Sse41);
        }
        if best >= Level::Avx2 {
            v.push(Level::Avx2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if best == Level::Neon {
        v.push(Level::Neon);
    }
    v
}

/// A small deterministic generator for the kernels' tests.
#[cfg(test)]
pub(crate) struct Rng(pub u64);

#[cfg(test)]
impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in `lo..=hi`.
    pub(crate) fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }
}
