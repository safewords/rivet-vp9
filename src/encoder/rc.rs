//! Rate control: a quantiser per frame for a target bitrate.
//!
//! The model is the usual one for a transform coder: a frame's size falls
//! as a power of the quantiser step, `bits = c * qstep^-s`, with `c` the
//! frame's complexity and `s` near 1. Each frame type (key, inter) keeps
//! the complexity of its last frame and a slope; a frame is coded at the
//! quantiser the model predicts for its budget, and recoded (at most
//! [`Config::max_recodes`] times) while its size misses the budget by more
//! than a tolerance, each recode refining the slope from the two sizes.
//!
//! One pass: the budget of a frame is the per-frame share of the bitrate,
//! a key frame [`KEY_BOOST`] times that, less the overspend so far spread
//! over the next second (or the frames left before the next key frame).
//! Two passes: a [`FirstPass`] codes every frame at one quantiser first; the
//! second pass divides what is left of the clip's budget among the frames
//! left by weight — the first-pass size to the power 0.8, times
//! [`KEY_BOOST`] for a key frame — and starts each frame's search from its
//! own first-pass complexity.

use super::{Config, Encoder};
use crate::frame::Frame;
use crate::tables::AC_QLOOKUP;
use crate::{Error, Result};

/// A key frame's budget in inter frame budgets (one pass), or its weight
/// over an inter frame of the same first-pass size (two passes).
pub const KEY_BOOST: f64 = 4.0;

/// A frame that refreshes GOLDEN: its budget in inter frame budgets (one
/// pass), or its weight over an inter frame of the same first-pass size
/// (two passes).
pub const GOLDEN_BOOST: f64 = 2.0;

/// A second pass weighs each frame by its first-pass size to this power.
const TWO_PASS_EXPONENT: f64 = 0.8;

/// How far (as a ratio) a frame may miss its budget before it is recoded.
const TOLERANCE: f64 = 0.12;

/// The quantiser of a [`FirstPass`].
const FIRST_PASS_Q: u8 = 96;

/// What a first pass measured: every frame's size at one quantiser.
#[derive(Debug, Clone, PartialEq)]
pub struct FirstPassStats {
    q: u8,
    /// Bits of each frame, and its boost (key, golden or 1).
    frames: Vec<(u64, f64)>,
}

impl FirstPassStats {
    /// Number of frames measured.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether no frame was measured.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

/// The first pass of two-pass rate control: give it every frame of the
/// clip, then put [`FirstPass::finish`]'s statistics in
/// [`Config::two_pass`] and encode the clip again with them.
///
/// ```
/// # let frames = vec![vp9::Frame::new(64, 64, 8, vp9::ChromaFormat::Yuv420); 3];
/// let mut cfg = vp9::Config::new(64, 64);
/// cfg.target_bitrate = Some(200_000);
/// let mut first = vp9::encoder::FirstPass::new(cfg.clone());
/// for f in &frames {
///     first.add(f)?;
/// }
/// cfg.two_pass = Some(first.finish());
/// let mut enc = vp9::Encoder::new(cfg);
/// for f in &frames {
///     let packet = enc.encode(f)?;
/// }
/// # Ok::<(), vp9::Error>(())
/// ```
pub struct FirstPass {
    enc: Encoder,
    stats: FirstPassStats,
}

impl FirstPass {
    /// A first pass for `cfg` (its rate control settings are not used: the
    /// first pass codes at a fixed quantiser).
    pub fn new(cfg: Config) -> Self {
        let mut c = cfg;
        c.target_bitrate = None;
        c.two_pass = None;
        c.quantizer = FIRST_PASS_Q;
        FirstPass {
            enc: Encoder::new(c),
            stats: FirstPassStats {
                q: FIRST_PASS_Q,
                frames: Vec::new(),
            },
        }
    }

    /// Codes the next frame of the clip and records its size.
    pub fn add(&mut self, frame: &Frame) -> Result<()> {
        let pkt = self.enc.encode(frame)?;
        let boost = if self.enc.last_was_key {
            KEY_BOOST
        } else if self.enc.refresh_golden {
            GOLDEN_BOOST
        } else {
            1.0
        };
        self.stats.frames.push((pkt.len() as u64 * 8, boost));
        Ok(())
    }

    /// The statistics for [`Config::two_pass`].
    pub fn finish(self) -> FirstPassStats {
        self.stats
    }
}

/// The rate controller's state between frames.
#[derive(Debug, Clone)]
pub(crate) struct RateCtl {
    /// Budget of an average frame, bits.
    per_frame: f64,
    /// Bits spent beyond the budgets so far (negative: saved).
    debt: f64,
    /// Bits spent in total, and frames coded.
    spent: f64,
    coded: usize,
    /// Per frame type (0 key, 1 inter): ln c of the last frame, and s.
    log_c: [Option<f64>; 2],
    slope: [f64; 2],
}

/// One attempt at coding a frame: the quantiser and the size.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Attempt {
    pub q: u8,
    pub bits: f64,
}

impl RateCtl {
    pub(crate) fn new(cfg: &Config) -> Self {
        let rate = cfg.target_bitrate.unwrap_or(0) as f64;
        RateCtl {
            per_frame: rate / cfg.frame_rate.max(1e-3),
            debt: 0.0,
            spent: 0.0,
            coded: 0,
            log_c: [None, None],
            slope: [0.9, 0.9],
        }
    }

    fn ln_qstep(cfg: &Config, q: u8) -> f64 {
        let bd = ((cfg.bit_depth - 8) >> 1) as usize;
        (AC_QLOOKUP[bd][q as usize] as f64).ln()
    }

    /// The budget of the next frame, bits.
    pub(crate) fn budget(&self, cfg: &Config, key: bool, golden: bool, frames_to_key: u64) -> f64 {
        if let Some(fp) = &cfg.two_pass
            && self.coded < fp.frames.len()
        {
            // A frame's weight: its first-pass size, compressed a little
            // (easy frames get relatively more), key and golden frames
            // boosted (later frames are predicted from them).
            let w = |f: &(u64, f64)| (f.0 as f64).powf(TWO_PASS_EXPONENT) * f.1;
            let total = self.per_frame * fp.frames.len() as f64;
            let left: f64 = fp.frames[self.coded..].iter().map(w).sum();
            let share = w(&fp.frames[self.coded]) / left.max(1.0);
            let nominal =
                total * w(&fp.frames[self.coded]) / fp.frames.iter().map(w).sum::<f64>().max(1.0);
            return ((total - self.spent) * share).clamp(0.25 * nominal, 4.0 * nominal);
        }
        // Over a golden interval the boosts are budget-neutral: the other
        // inter frames give up what the golden frame gets.
        let g = cfg.golden_interval as f64;
        let inter = if g > 0.0 {
            self.per_frame * g / (g - 1.0 + GOLDEN_BOOST)
        } else {
            self.per_frame
        };
        let base = if key {
            self.per_frame * KEY_BOOST
        } else if golden {
            inter * GOLDEN_BOOST
        } else {
            inter
        };
        let window = (frames_to_key as f64).min(cfg.frame_rate.round()).max(4.0);
        (base - self.debt / window).clamp(0.25 * base, 4.0 * base)
    }

    /// ln c to start the search from: the frame's own first-pass
    /// complexity in a second pass, else the last frame of its type's.
    fn start_log_c(&self, cfg: &Config, key: bool) -> Option<f64> {
        let t = (!key) as usize;
        if let Some(fp) = &cfg.two_pass
            && let Some(&(bits, _)) = fp.frames.get(self.coded)
        {
            return Some((bits.max(8) as f64).ln() + self.slope[t] * Self::ln_qstep(cfg, fp.q));
        }
        self.log_c[t]
    }

    /// The quantiser the model gives for `target` bits.
    fn q_for(cfg: &Config, log_c: f64, slope: f64, target: f64) -> u8 {
        let (lo, hi) = (cfg.min_quantizer.max(1), cfg.max_quantizer.max(1));
        let lt = target.max(1.0).ln();
        (lo..=hi.max(lo))
            .min_by(|&a, &b| {
                let da = (log_c - slope * Self::ln_qstep(cfg, a) - lt).abs();
                let db = (log_c - slope * Self::ln_qstep(cfg, b) - lt).abs();
                da.total_cmp(&db)
            })
            .unwrap_or(lo)
    }

    /// The first quantiser to try for a frame with budget `target`.
    pub(crate) fn first_q(&self, cfg: &Config, key: bool, target: f64) -> u8 {
        match self.start_log_c(cfg, key) {
            Some(lc) => Self::q_for(cfg, lc, self.slope[(!key) as usize], target),
            // Nothing known: the middle of the range; recoding finds it.
            None => cfg
                .quantizer
                .clamp(cfg.min_quantizer.max(1), cfg.max_quantizer.max(1)),
        }
    }

    /// After `tries` (the last one newest), the quantiser to try next, or
    /// `None` to stop. Updates the model of the frame type.
    pub(crate) fn next_q(
        &mut self,
        cfg: &Config,
        key: bool,
        target: f64,
        tries: &[Attempt],
        max_tries: usize,
    ) -> Option<u8> {
        let t = (!key) as usize;
        let a = *tries.last()?;
        // The slope from the last two attempts at different quantisers.
        if let Some(b) = tries.iter().rev().skip(1).find(|b| b.q != a.q) {
            let dq = Self::ln_qstep(cfg, a.q) - Self::ln_qstep(cfg, b.q);
            if dq.abs() > 1e-6 {
                let s = -(a.bits.max(8.0).ln() - b.bits.max(8.0).ln()) / dq;
                self.slope[t] = s.clamp(0.4, 2.5);
            }
        }
        let lc = a.bits.max(8.0).ln() + self.slope[t] * Self::ln_qstep(cfg, a.q);
        self.log_c[t] = Some(lc);
        let miss = a.bits / target.max(1.0);
        if tries.len() >= max_tries || (miss - 1.0).abs() <= TOLERANCE {
            return None;
        }
        let q = Self::q_for(cfg, lc, self.slope[t], target);
        // Stop at a quantiser already tried, or when the search cannot
        // move in the direction it needs.
        if tries.iter().any(|x| x.q == q) || (miss > 1.0 && q <= a.q) || (miss < 1.0 && q >= a.q) {
            return None;
        }
        Some(q)
    }

    /// Records the size of the frame as sent.
    pub(crate) fn commit(&mut self, target: f64, bits: f64) {
        self.debt += bits - target;
        self.spent += bits;
        self.coded += 1;
    }
}

/// Checks the rate control settings.
pub(crate) fn validate(cfg: &Config) -> Result<()> {
    if let Some(r) = cfg.target_bitrate {
        if r == 0 {
            return Err(Error::invalid("target_bitrate must be above 0"));
        }
        if !(cfg.frame_rate.is_finite() && cfg.frame_rate > 0.0) {
            return Err(Error::invalid("frame_rate must be above 0"));
        }
        if cfg.min_quantizer > cfg.max_quantizer {
            return Err(Error::invalid("min_quantizer is above max_quantizer"));
        }
    }
    Ok(())
}
