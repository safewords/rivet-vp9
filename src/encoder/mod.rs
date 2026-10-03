//! A VP9 encoder: profiles 0 to 3 (8, 10 and 12 bits; 4:2:0, 4:2:2, 4:4:0
//! and 4:4:4), a fixed quantiser.
//!
//! The profile follows from [`Config::bit_depth`] and [`Config::chroma`]
//! ([`Config::profile`]). Every frame is one packet. Key frames code each
//! block with the best of the ten intra modes; inter frames add
//! single-reference prediction from LAST (the previous frame) or GOLDEN (an
//! older frame, coded finer) — motion search to quarter-sample precision,
//! NEARESTMV / NEARMV / ZEROMV / NEWMV — with intra as the alternative. The
//! partition and transform size are searched by rate-distortion cost
//! ([`Config::speed`] 0 and 1), or fixed (speed 2: square blocks of
//! [`Config::block_size`], smaller where the frame edge forces it, the
//! largest transform that fits), and quantisation is plain rounding with a
//! dead zone. The coefficient
//! probabilities that pay for their update are sent in each frame's
//! compressed header (judged on a first coding pass); the others stay at the
//! specification's defaults, and there is no backward adaptation (inter
//! frames are error resilient, so nothing carries over between frames but
//! the reference pictures). `quantizer` 0 is lossless (4x4 Walsh-Hadamard).
//!
//! The encoder reconstructs with the decoder's own prediction and inverse
//! transform code, and keeps its reference frames by decoding its own
//! output with an internal [`Decoder`]: what it predicts from is, by
//! construction, what a decoder will have, and [`Encoder::reconstruction`]
//! hands it out. Debug builds also loop filter the encoder's own
//! reconstruction and compare it with the decoded packet sample by sample.
//!
//! # Rate control
//!
//! Without a [`Config::target_bitrate`] every frame is coded at
//! [`Config::quantizer`]. With one, each frame gets a budget and is coded at
//! the quantiser a model of its size predicts (`bits = c * qstep^-s`, per
//! frame type, refined from every frame coded), then coded again — up to
//! [`Config::max_recodes`] times — while it misses the budget by more than
//! 12%; the attempt nearest the budget is sent.
//!
//! - **One pass**: a frame's budget is the bitrate's share per frame (a key
//!   frame [`KEY_BOOST`] times that, a frame that refreshes GOLDEN
//!   [`GOLDEN_BOOST`] times), less the overspend so far spread over
//!   the next second or the frames before the next key frame.
//! - **Two passes**: a [`FirstPass`] codes the clip at one quantiser; with
//!   its statistics in [`Config::two_pass`], what is left of the clip's
//!   budget is divided among the frames left by weight — the first-pass
//!   size to the power 0.8, key frames times [`KEY_BOOST`], golden frames
//!   times [`GOLDEN_BOOST`] — and each
//!   frame's search starts from its own first-pass complexity.

mod fdct;
mod rc;
mod tile;

pub use rc::{FirstPass, FirstPassStats, GOLDEN_BOOST, KEY_BOOST};

use std::sync::Arc;

use crate::bits::BitWriter;
use crate::bool_coder::{BoolEncoder, Sink};
use crate::consts::*;
use crate::decoder::{Decoder, FrameDec, RefFrame};
use crate::frame::{ChromaFormat, ColorSpace, Frame};
use crate::header::{FrameHeader, Segmentation};
use crate::probs::{Counts, Probs};
use crate::{Error, Result};

/// Encoder settings.
#[derive(Debug, Clone)]
pub struct Config {
    /// Width in pixels (1 to 65536).
    pub width: u32,
    /// Height in pixels (1 to 65536).
    pub height: u32,
    /// The quantiser index `base_q_idx`, 0 to 255: lower is better quality.
    /// 0 codes losslessly. With a [`Config::target_bitrate`], the
    /// quantiser the first frame is tried at.
    pub quantizer: u8,
    /// Rate control: a target in bits per second (at
    /// [`Config::frame_rate`]) instead of the fixed [`Config::quantizer`].
    /// See [Rate control](self#rate-control).
    pub target_bitrate: Option<u64>,
    /// Frames per second, which turns the bitrate into bits per frame.
    pub frame_rate: f64,
    /// The quantiser range rate control may use (1 to 255; never
    /// lossless).
    pub min_quantizer: u8,
    /// See [`Config::min_quantizer`].
    pub max_quantizer: u8,
    /// How many times rate control may code a frame again at another
    /// quantiser when it misses its budget by more than 12%.
    pub max_recodes: u32,
    /// Two-pass rate control: the statistics of a [`FirstPass`] over the
    /// same frames.
    pub two_pass: Option<FirstPassStats>,
    /// A key frame every this many frames; 1 makes every frame a key frame.
    pub keyframe_interval: u32,
    /// Loop filter level 0 to 63; `None` derives it from the quantiser.
    pub loop_filter_level: Option<u8>,
    /// Block size of the fixed partition: 8, 16, 32 or 64 (used at
    /// [`Config::speed`] 2; the slower speeds search the partition).
    pub block_size: u32,
    /// Speed against compression: 0 searches the partition (every
    /// partition type, 64x64 down to 8x8) and every transform size; 1 (the
    /// default) the partition with NONE and SPLIT (HORZ / VERT only where
    /// the frame edge forces them) and the two largest transform sizes; 2
    /// codes the fixed partition of [`Config::block_size`] with the largest
    /// transform that fits.
    pub speed: u8,
    /// Inter frames search two references: LAST (the previous frame) and
    /// GOLDEN, the last key frame, replaced by every this-many-th frame
    /// after it (default 8), which is coded finer: at 3/4 of
    /// [`Config::quantizer`], or with [`GOLDEN_BOOST`] times an inter
    /// frame's budget. 0 searches LAST only.
    pub golden_interval: u32,
    /// Motion search range in whole pixels.
    pub search_range: u32,
    /// Colour space to signal.
    pub color_space: ColorSpace,
    /// Signal full-range (instead of studio-range) samples.
    pub full_range: bool,
    /// Bits per sample of the frames to encode: 8, 10 or 12.
    pub bit_depth: u32,
    /// Chroma sampling of the frames to encode.
    pub chroma: ChromaFormat,
}

impl Config {
    /// Defaults for a `width` x `height` stream: quantiser 64, a key frame
    /// every 60 frames, 16x16 blocks, +-16 pixel motion search.
    pub fn new(width: u32, height: u32) -> Self {
        Config {
            width,
            height,
            quantizer: 64,
            target_bitrate: None,
            frame_rate: 30.0,
            min_quantizer: 1,
            max_quantizer: 255,
            max_recodes: 2,
            two_pass: None,
            keyframe_interval: 60,
            loop_filter_level: None,
            block_size: 16,
            speed: 1,
            search_range: 16,
            golden_interval: 8,
            color_space: ColorSpace::Bt601,
            full_range: false,
            bit_depth: 8,
            chroma: ChromaFormat::Yuv420,
        }
    }

    /// The VP9 profile the settings need: 0 for 8-bit 4:2:0, 1 for 8-bit
    /// 4:2:2 / 4:4:0 / 4:4:4, 2 for 10 / 12-bit 4:2:0, 3 for 10 / 12-bit
    /// with the other chroma formats.
    pub fn profile(&self) -> u8 {
        let high = self.bit_depth > 8;
        let full = self.chroma != ChromaFormat::Yuv420;
        (high as u8) << 1 | full as u8
    }

    /// Checks the settings; [`Encoder::encode`] refuses to encode with an
    /// invalid configuration and returns this error.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.width > 65536 || self.height > 65536 {
            return Err(Error::invalid(
                "frame size must be 1 to 65536 in each direction",
            ));
        }
        if ![8, 16, 32, 64].contains(&self.block_size) {
            return Err(Error::invalid("block_size must be 8, 16, 32 or 64"));
        }
        rc::validate(self)?;
        if ![8, 10, 12].contains(&self.bit_depth) {
            return Err(Error::invalid("bit_depth must be 8, 10 or 12"));
        }
        if self.color_space == ColorSpace::Rgb && self.chroma != ChromaFormat::Yuv444 {
            return Err(Error::invalid("sRGB (planes G, B, R) is coded 4:4:4 only"));
        }
        Ok(())
    }
}

/// A VP9 encoder. See the [module documentation](self).
pub struct Encoder {
    cfg: Config,
    frames: u64,
    force_key: bool,
    /// Decodes every packet produced: the reference frames.
    dec: Decoder,
    /// The last frame's reconstruction (the decoded packet).
    recon: Option<Frame>,
    /// Whether the last frame was a key frame, and its quantiser.
    last_was_key: bool,
    last_q: u8,
    rc: rc::RateCtl,
    /// Frames since the last key frame, and whether this one refreshes
    /// GOLDEN.
    since_key: u64,
    refresh_golden: bool,
}

/// The LAST, GOLDEN and ALTREF frames.
type Refs = [Option<Arc<RefFrame>>; 3];

/// A frame coded at one quantiser, not yet sent.
struct Coded {
    packet: Vec<u8>,
    /// Debug builds: the encoder's reconstruction, loop filtered.
    recon: Option<[crate::decoder::PlaneBuf; 3]>,
    q: u8,
}

impl Encoder {
    /// An encoder for `cfg`. The configuration is checked by the first
    /// [`Encoder::encode`] (or earlier with [`Config::validate`]).
    pub fn new(cfg: Config) -> Self {
        let mut dec = Decoder::new();
        dec.set_max_pixels(cfg.width as u64 * cfg.height as u64);
        Encoder {
            frames: 0,
            force_key: true,
            rc: rc::RateCtl::new(&cfg),
            dec,
            recon: None,
            last_was_key: false,
            last_q: 0,
            since_key: 0,
            refresh_golden: false,
            cfg,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Makes the next frame a key frame.
    pub fn force_keyframe(&mut self) {
        self.force_key = true;
    }

    /// The reconstruction of the last frame encoded: exactly what a decoder
    /// outputs for the last packet (the encoder decodes its own packets to
    /// keep its references). `None` before the first frame.
    pub fn reconstruction(&self) -> Option<&Frame> {
        self.recon.as_ref()
    }

    /// The quantiser index (`base_q_idx`) the last frame was coded with.
    pub fn last_quantizer(&self) -> u8 {
        self.last_q
    }

    /// Whether the last frame was coded as a key frame.
    pub fn last_was_keyframe(&self) -> bool {
        self.last_was_key
    }

    /// Encodes one frame into one packet (a complete VP9 frame).
    pub fn encode(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        self.cfg.validate()?;
        if frame.width != self.cfg.width || frame.height != self.cfg.height {
            return Err(Error::invalid(format!(
                "frame is {}x{}, the encoder was configured for {}x{}",
                frame.width, frame.height, self.cfg.width, self.cfg.height
            )));
        }
        if frame.bit_depth != self.cfg.bit_depth || frame.chroma != self.cfg.chroma {
            return Err(Error::invalid(format!(
                "frame is {}-bit {:?}, the encoder was configured for {}-bit {:?}",
                frame.bit_depth, frame.chroma, self.cfg.bit_depth, self.cfg.chroma
            )));
        }
        let interval = self.cfg.keyframe_interval.max(1) as u64;
        let key = self.force_key || self.frames.is_multiple_of(interval);
        let key = key || self.dec.ref_slot(0).is_none();
        // LAST, GOLDEN and ALTREF are slots 0, 1 and 2.
        let refs: Refs = if key {
            Default::default()
        } else {
            [0, 1, 2].map(|i| self.dec.ref_slot(i))
        };
        self.since_key = if key { 0 } else { self.since_key + 1 };
        let g = self.cfg.golden_interval as u64;
        self.refresh_golden = !key && g > 0 && self.since_key.is_multiple_of(g);
        let src = tile::Source::new(frame, &self.header(key, 0));
        let coded = if self.cfg.target_bitrate.is_some() {
            self.rate_controlled(&src, key, &refs, interval)?
        } else {
            // A frame that becomes GOLDEN is predicted from for the next
            // golden_interval frames: it is coded finer.
            let q = self.cfg.quantizer;
            let q = if self.refresh_golden && q > 0 {
                ((q as u32 * 3) / 4).max(1) as u8
            } else {
                q
            };
            self.code_frame(&src, key, &refs, q)?
        };
        let Coded { packet, recon, q } = coded;
        // Keep the references a decoder will have.
        let out = self.dec.decode(&packet).map_err(|e| {
            Error::invalid(format!(
                "internal error: the encoder's own frame does not decode: {e}"
            ))
        })?;
        // Debug builds check that the encoder reconstructed exactly what the
        // decoder decodes: its own prediction and residual, loop filtered
        // with the decoder's filter, against the decoded packet.
        if let (Some(planes), Some(out)) = (recon, &out) {
            for (p, buf) in planes.iter().enumerate() {
                let pl = out.planes[p];
                for y in 0..pl.height {
                    for x in 0..pl.width {
                        assert_eq!(
                            buf.data[y as usize * buf.stride + x as usize],
                            out.sample(p, x, y),
                            "encoder/decoder mismatch, plane {p} at ({x}, {y}), frame {}",
                            self.frames
                        );
                    }
                }
            }
        }
        self.recon = out;
        self.last_was_key = key;
        self.last_q = q;
        self.frames += 1;
        self.force_key = false;
        Ok(packet)
    }

    /// Codes the frame at the quantiser rate control picks for its budget,
    /// recoding while it misses; the attempt nearest the budget is sent.
    fn rate_controlled(
        &mut self,
        src: &tile::Source,
        key: bool,
        refs: &Refs,
        interval: u64,
    ) -> Result<Coded> {
        let to_key = interval - self.frames % interval;
        let target = self.rc.budget(&self.cfg, key, self.refresh_golden, to_key);
        let mut q = self.rc.first_q(&self.cfg, key, target);
        // With nothing known about the content, the first frame may take
        // a few more attempts.
        let extra = if self.frames == 0 && self.cfg.two_pass.is_none() {
            3
        } else {
            0
        };
        let max_tries = 1 + self.cfg.max_recodes as usize + extra;
        let mut tries = Vec::new();
        let mut best: Option<Coded> = None;
        let miss = |bits: f64| (bits.max(1.0) / target.max(1.0)).ln().abs();
        loop {
            let c = self.code_frame(src, key, refs, q)?;
            let bits = c.packet.len() as f64 * 8.0;
            tries.push(rc::Attempt { q, bits });
            if best
                .as_ref()
                .is_none_or(|b| miss(bits) < miss(b.packet.len() as f64 * 8.0))
            {
                best = Some(c);
            }
            match self.rc.next_q(&self.cfg, key, target, &tries, max_tries) {
                Some(n) => q = n,
                None => break,
            }
        }
        let best = best.expect("one attempt at least");
        self.rc.commit(target, best.packet.len() as f64 * 8.0);
        Ok(best)
    }

    /// Codes the frame at quantiser `q`: the packet, nothing committed.
    fn code_frame(&self, src: &tile::Source, key: bool, refs: &Refs, q: u8) -> Result<Coded> {
        let h = self.header(key, q);
        // A first pass with the default probabilities measures the
        // coefficient statistics; the coefficient probabilities that pay
        // for their own update are sent, and the frame coded again with them.
        let defaults = Probs::default();
        // The second pass replays the first's block decisions: they do not
        // depend on the probabilities.
        let first = self.encode_pass(&h, src, refs, &defaults, None);
        let probs = updated_coef_probs(&defaults, &first.stats, h.tx_mode);
        let second = self.encode_pass(&h, src, refs, &probs, Some(first.decisions));
        let comp = compressed_header(&h, &defaults, &probs);
        let mut w = BitWriter::default();
        self.uncompressed_header(&mut w, &h, comp.len())?;
        let mut packet = w.finish();
        packet.extend_from_slice(&comp);
        packet.extend_from_slice(&second.tiles);
        Ok(Coded {
            packet,
            recon: second.recon,
            q,
        })
    }

    /// Codes the tiles of a frame with `probs`; returns the tile data, the
    /// coefficient statistics and (debug builds, a replaying pass) the
    /// loop-filtered reconstruction.
    fn encode_pass(
        &self,
        h: &FrameHeader,
        src: &tile::Source,
        refs: &Refs,
        probs: &Probs,
        replay: Option<Vec<tile::Decision>>,
    ) -> Pass {
        let keep = replay.is_some();
        let seg = Segmentation::default();
        let mut probs = probs.clone();
        let mut counts = Box::<Counts>::default();
        // The references searched: LAST, and GOLDEN when it is another
        // picture.
        let mut search: Vec<(i8, &RefFrame)> = Vec::new();
        if let Some(last) = &refs[0] {
            search.push((LAST_FRAME, last));
            if let Some(golden) = &refs[1]
                && self.cfg.golden_interval > 0
                && !Arc::ptr_eq(golden, last)
            {
                search.push((GOLDEN_FRAME, golden));
            }
        }
        let mut fd = FrameDec::new(h, &seg, &mut probs, &mut counts, &[], None, refs.clone());
        let mut te = tile::TileEncoder::new(&self.cfg, h, src, &search);
        te.replay = replay.map(|v| (v, 0));
        let t = te.encode_tiles(&mut fd);
        let recon = if keep && cfg!(debug_assertions) {
            let (mut planes, mi) = fd.finish();
            let lf = crate::header::LoopFilter {
                level: self.loop_filter_level(h.base_q_idx as u8),
                sharpness: 0,
                delta_enabled: false,
                ..Default::default()
            };
            if lf.level != 0 {
                crate::decoder::loopfilter::filter_frame(h, &lf, &seg, &mi, &mut planes);
            }
            Some(planes)
        } else {
            None
        };
        Pass {
            tiles: t,
            stats: te.stats,
            decisions: te.decisions,
            recon,
        }
    }

    fn loop_filter_level(&self, q: u8) -> u8 {
        if let Some(l) = self.cfg.loop_filter_level {
            return l.min(63);
        }
        let q = q as u32;
        if q == 0 {
            0
        } else {
            ((q * 10 + 32) / 64).min(63) as u8
        }
    }

    /// The header both the writer and the reconstruction use.
    fn header(&self, key: bool, q: u8) -> FrameHeader {
        let (w, ht) = (self.cfg.width, self.cfg.height);
        let mi_cols = w.div_ceil(8);
        let mi_rows = ht.div_ceil(8);
        let sb64_cols = mi_cols.div_ceil(8);
        let sb64_rows = mi_rows.div_ceil(8);
        let mut min_log2 = 0;
        while (MAX_TILE_WIDTH_B64 << min_log2) < sb64_cols {
            min_log2 += 1;
        }
        let lossless = q == 0;
        let (ss_x, ss_y) = self.cfg.chroma.shifts();
        FrameHeader {
            profile: self.cfg.profile(),
            frame_type: if key { KEY_FRAME } else { 1 },
            last_frame_type: KEY_FRAME,
            show_frame: true,
            error_resilient_mode: !key,
            bit_depth: self.cfg.bit_depth,
            color_space: self.cfg.color_space,
            color_range: self.cfg.full_range || self.cfg.color_space == ColorSpace::Rgb,
            subsampling_x: ss_x,
            subsampling_y: ss_y,
            refresh_frame_flags: if key {
                0xff
            } else {
                0x01 | (self.refresh_golden as u8) << 1
            },
            ref_frame_idx: [0, 1, 2],
            width: w,
            height: ht,
            render_width: w,
            render_height: ht,
            interpolation_filter: EIGHTTAP,
            frame_parallel_decoding_mode: true,
            base_q_idx: q as i32,
            lossless,
            tile_cols_log2: min_log2,
            frame_is_intra: key,
            mi_cols,
            mi_rows,
            sb64_cols,
            sb64_rows,
            tx_mode: if lossless {
                ONLY_4X4
            } else if self.cfg.speed <= 1 {
                TX_MODE_SELECT
            } else {
                ALLOW_32X32
            },
            reference_mode: SINGLE_REFERENCE,
            ..FrameHeader::default()
        }
    }

    fn uncompressed_header(
        &self,
        w: &mut BitWriter,
        h: &FrameHeader,
        comp_len: usize,
    ) -> Result<()> {
        if comp_len >= 1 << 16 {
            return Err(Error::unsupported("compressed header above 64 KB"));
        }
        let key = h.frame_type == KEY_FRAME;
        w.f(2, 2); // frame_marker
        w.f(1, h.profile as u32 & 1); // profile_low_bit
        w.f(1, h.profile as u32 >> 1); // profile_high_bit
        if h.profile == 3 {
            w.f(1, 0); // reserved_zero
        }
        w.f(1, 0); // show_existing_frame
        w.f(1, h.frame_type as u32);
        w.f(1, 1); // show_frame
        w.f(1, h.error_resilient_mode as u32);
        if key {
            w.f(8, 0x49);
            w.f(8, 0x83);
            w.f(8, 0x42);
            // color_config()
            if h.profile >= 2 {
                w.f(1, (h.bit_depth == 12) as u32); // ten_or_twelve_bit
            }
            w.f(3, h.color_space.bits());
            if h.color_space != ColorSpace::Rgb {
                w.f(1, h.color_range as u32);
                if h.profile == 1 || h.profile == 3 {
                    w.f(1, h.subsampling_x);
                    w.f(1, h.subsampling_y);
                    w.f(1, 0); // reserved_zero
                }
            } else if h.profile == 1 || h.profile == 3 {
                w.f(1, 0); // reserved_zero
            }
            w.f(16, h.width - 1);
            w.f(16, h.height - 1);
            w.f(1, 0); // render_and_frame_size_different
        } else {
            // show_frame is 1, so intra_only is not coded; error resilient,
            // so reset_frame_context is not either.
            w.f(8, h.refresh_frame_flags as u32);
            for i in 0..3 {
                w.f(3, h.ref_frame_idx[i] as u32);
                w.f(1, 0); // ref_frame_sign_bias
            }
            w.f(1, 1); // found_ref: the size of LAST
            w.f(1, 0); // render_and_frame_size_different
            w.f(1, h.allow_high_precision_mv as u32);
            w.f(1, 0); // is_filter_switchable
            w.f(2, 1); // raw_interpolation_filter: literal_to_type[1] = EIGHTTAP
        }
        if !h.error_resilient_mode {
            w.f(1, 0); // refresh_frame_context
            w.f(1, 1); // frame_parallel_decoding_mode
        }
        w.f(2, 0); // frame_context_idx
        // loop_filter_params()
        w.f(6, self.loop_filter_level(h.base_q_idx as u8) as u32);
        w.f(3, 0); // sharpness
        w.f(1, 0); // loop_filter_delta_enabled
        // quantization_params()
        w.f(8, h.base_q_idx as u32);
        w.f(1, 0);
        w.f(1, 0);
        w.f(1, 0);
        w.f(1, 0); // segmentation_enabled
        // tile_info(): the minimum number of tile columns, one tile row.
        let mut max_log2 = 1;
        while (h.sb64_cols >> max_log2) >= MIN_TILE_WIDTH_B64 {
            max_log2 += 1;
        }
        max_log2 -= 1;
        if h.tile_cols_log2 < max_log2 {
            w.f(1, 0); // increment_tile_cols_log2
        }
        w.f(1, 0); // tile_rows_log2
        w.f(16, comp_len as u32);
        Ok(())
    }
}

/// What one coding pass over a frame produced.
struct Pass {
    tiles: Vec<u8>,
    stats: Box<tile::CoefStats>,
    decisions: Vec<tile::Decision>,
    /// Debug builds, second pass: the reconstruction, loop filtered.
    recon: Option<[crate::decoder::PlaneBuf; 3]>,
}

/// Bits of a bool of probability `p` (of a 0): its cost when 0 and when 1.
fn bool_cost(p: u8) -> (f64, f64) {
    let p0 = p as f64 / 256.0;
    (-p0.log2(), -(1.0 - p0).log2())
}

/// The deltaProb whose inv_remap_prob takes `old` to `new` with the
/// fewest bits, and those bits (the code of decode_term_subexp).
fn best_delta(old: u8, new: u8) -> Option<(u32, f64)> {
    (0..254u32)
        .filter(|&d| crate::header::inv_remap_prob(d, old) == new)
        .map(|d| (d, subexp_bits(d)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

fn subexp_bits(d: u32) -> f64 {
    match d {
        0..=15 => 5.0,
        16..=31 => 6.0,
        32..=63 => 8.0,
        64..=128 => 10.0,
        _ => 11.0,
    }
}

/// The inverse of decode_term_subexp (6.3.4).
fn write_subexp(e: &mut BoolEncoder, d: u32) {
    if d < 16 {
        e.literal(1, 0);
        e.literal(4, d);
    } else if d < 32 {
        e.literal(2, 0b10);
        e.literal(4, d - 16);
    } else if d < 64 {
        e.literal(3, 0b110);
        e.literal(5, d - 32);
    } else {
        e.literal(3, 0b111);
        if d - 64 < 65 {
            e.literal(7, d - 64);
        } else {
            // v of 65 or more, then one more bit: d = (v << 1) - 1 + bit.
            let bit = (d + 1) & 1;
            e.literal(7, (d + 1 - bit) >> 1);
            e.literal(1, bit);
        }
    }
}

/// The inverse of diff_update_prob (6.3.3).
fn write_diff_update(e: &mut BoolEncoder, old: u8, new: u8) {
    match best_delta(old, new) {
        Some((d, _)) if old != new => {
            e.write(true, 252);
            write_subexp(e, d);
        }
        _ => e.write(false, 252),
    }
}

#[allow(clippy::needless_range_loop)]
/// The coefficient probabilities worth updating, given how often each of
/// the first three nodes of every context saw a 0 and a 1. A transform
/// size's probabilities are updated together or not at all, as its
/// update_probs flag decides.
fn updated_coef_probs(old: &Probs, stats: &tile::CoefStats, tx_mode: u8) -> Probs {
    let mut p = old.clone();
    let (no0, _) = bool_cost(252);
    let (_, yes1) = bool_cost(252);
    let max_tx = TX_MODE_TO_BIGGEST_TX_SIZE[tx_mode as usize] as usize;
    for tx in 0..=max_tx {
        let mut saving = -1.0; // the update_probs flag
        let mut cand = p.coef[tx];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..6 {
                    for l in 0..if k == 0 { 3 } else { 6 } {
                        for m in 0..3 {
                            let [c0, c1] = stats[tx][i][j][k][l][m];
                            let o = old.coef[tx][i][j][k][l][m];
                            let cost = |q: u8| {
                                let (z, n) = bool_cost(q);
                                c0 as f64 * z + c1 as f64 * n
                            };
                            let mut best = (o, cost(o) + no0);
                            if c0 + c1 > 0 {
                                let n = (c0 + c1) as u64;
                                let target = ((c0 as u64 * 256 + n / 2) / n).clamp(1, 255) as u8;
                                for q in [
                                    target,
                                    target.saturating_sub(2).max(1),
                                    target.saturating_add(2),
                                ] {
                                    if q != o
                                        && let Some((_, bits)) = best_delta(o, q)
                                    {
                                        let c = cost(q) + yes1 + bits;
                                        if c < best.1 {
                                            best = (q, c);
                                        }
                                    }
                                }
                            }
                            cand[i][j][k][l][m] = best.0;
                            saving += cost(o) + no0 - best.1;
                        }
                    }
                }
            }
        }
        if saving > 0.0 {
            p.coef[tx] = cand;
        }
    }
    p
}

/// compressed_header(): the transform mode, the coefficient probability
/// updates from `old` to `new`, and no other updates.
fn compressed_header(h: &FrameHeader, old: &Probs, new: &Probs) -> Vec<u8> {
    let mut e = BoolEncoder::new();
    let no = |e: &mut BoolEncoder| e.write(false, 252);
    if !h.lossless {
        e.literal(2, ALLOW_32X32 as u32);
        e.literal(1, (h.tx_mode == TX_MODE_SELECT) as u32); // tx_mode_select
        if h.tx_mode == TX_MODE_SELECT {
            // tx_mode_probs(): no updates (8x8: 1 per context, 16x16: 2,
            // 32x32: 3; two contexts each).
            for _ in 0..2 * (1 + 2 + 3) {
                no(&mut e);
            }
        }
    }
    // read_coef_probs()
    let max_tx = TX_MODE_TO_BIGGEST_TX_SIZE[h.tx_mode as usize] as usize;
    for tx in 0..=max_tx {
        let update = old.coef[tx] != new.coef[tx];
        e.literal(1, update as u32);
        if update {
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..6 {
                        for l in 0..if k == 0 { 3 } else { 6 } {
                            for m in 0..3 {
                                let (a, b) =
                                    (old.coef[tx][i][j][k][l][m], new.coef[tx][i][j][k][l][m]);
                                write_diff_update(&mut e, a, b);
                            }
                        }
                    }
                }
            }
        }
    }
    for _ in 0..3 {
        no(&mut e); // skip probs
    }
    if !h.frame_is_intra {
        for _ in 0..7 * 3 {
            no(&mut e); // inter_mode_probs
        }
        // interpolation_filter is not SWITCHABLE: no interp_filter_probs.
        for _ in 0..4 {
            no(&mut e); // is_inter_probs
        }
        // All sign biases equal: no compound prediction, nothing coded.
        for _ in 0..5 * 2 {
            no(&mut e); // single_ref_probs
        }
        for _ in 0..4 * 9 {
            no(&mut e); // y_mode_probs
        }
        for _ in 0..16 * 3 {
            no(&mut e); // partition_probs
        }
        // mv_probs(): joints, per component sign, classes, class0 bit,
        // bits, then class0_fr and fr; no hp (allow_high_precision_mv 0).
        for _ in 0..3 + 2 * (1 + 10 + 1 + 10) + 2 * (2 * 3 + 3) {
            no(&mut e);
        }
    }
    e.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: u32, h: u32, t: u32) -> Frame {
        let mut f = Frame::new(w, h, 8, ChromaFormat::Yuv420);
        for p in 0..3 {
            let pl = f.planes[p];
            for y in 0..pl.height {
                for x in 0..pl.width {
                    let v = if p == 0 {
                        ((x * 3 + y * 2 + t * 4) % 256) as u16 / 2
                            + 40
                            + (((x / 7 + y / 5 + t) % 3) * 20) as u16
                    } else {
                        (100 + (x + 2 * y + t) % 50) as u16
                    };
                    f.set_sample(p, x, y, v);
                }
            }
        }
        f
    }

    #[test]
    fn key_frame_decodes() {
        let f = gradient(80, 56, 0);
        let mut enc = Encoder::new(Config::new(80, 56));
        let pkt = enc.encode(&f).unwrap();
        let out = Decoder::new().decode(&pkt).unwrap().unwrap();
        assert_eq!((out.width, out.height), (80, 56));
    }

    #[test]
    fn lossless_round_trip() {
        let f = gradient(40, 24, 1);
        let mut cfg = Config::new(40, 24);
        cfg.quantizer = 0;
        let mut enc = Encoder::new(cfg);
        let mut dec = Decoder::new();
        for _ in 0..3 {
            let pkt = enc.encode(&f).unwrap();
            let out = dec.decode(&pkt).unwrap().unwrap();
            assert_eq!(out.data, f.data);
        }
    }
}
