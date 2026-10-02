//! A VP9 encoder: profile 0 (8-bit 4:2:0), a fixed quantiser.
//!
//! Every frame is one packet. Key frames code each block with the best of
//! the ten intra modes; inter frames add single-reference prediction from
//! the previous frame (motion search to quarter-sample precision, NEARESTMV
//! / NEARMV / ZEROMV / NEWMV) with intra as the alternative. The partition
//! is fixed (square blocks of [`Config::block_size`], smaller where the
//! frame edge forces it), the transform is the largest that fits the block,
//! and quantisation is plain rounding with a dead zone. Probabilities stay
//! at the specification's defaults: no forward or backward updates (inter
//! frames are error resilient, so nothing carries over between frames but
//! the reference pictures). `quantizer` 0 is lossless (4x4 Walsh-Hadamard).
//!
//! The encoder reconstructs with the decoder's own prediction and inverse
//! transform code, and keeps its reference frames by decoding its own
//! output with an internal [`Decoder`]: what it predicts from is, by
//! construction, what a decoder will have.

mod fdct;
mod tile;

use std::sync::Arc;

use crate::bits::BitWriter;
use crate::bool_coder::BoolEncoder;
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
    /// 0 codes losslessly.
    pub quantizer: u8,
    /// A key frame every this many frames; 1 makes every frame a key frame.
    pub keyframe_interval: u32,
    /// Loop filter level 0 to 63; `None` derives it from the quantiser.
    pub loop_filter_level: Option<u8>,
    /// Block size of the fixed partition: 8, 16, 32 or 64.
    pub block_size: u32,
    /// Motion search range in whole pixels.
    pub search_range: u32,
    /// Colour space to signal.
    pub color_space: ColorSpace,
    /// Signal full-range (instead of studio-range) samples.
    pub full_range: bool,
}

impl Config {
    /// Defaults for a `width` x `height` stream: quantiser 64, a key frame
    /// every 60 frames, 16x16 blocks, +-16 pixel motion search.
    pub fn new(width: u32, height: u32) -> Self {
        Config {
            width,
            height,
            quantizer: 64,
            keyframe_interval: 60,
            loop_filter_level: None,
            block_size: 16,
            search_range: 16,
            color_space: ColorSpace::Bt601,
            full_range: false,
        }
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
        if self.color_space == ColorSpace::Rgb {
            return Err(Error::unsupported("sRGB needs profile 1"));
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
}

impl Encoder {
    /// An encoder for `cfg`. The configuration is checked by the first
    /// [`Encoder::encode`] (or earlier with [`Config::validate`]).
    pub fn new(cfg: Config) -> Self {
        let mut dec = Decoder::new();
        dec.set_max_pixels(cfg.width as u64 * cfg.height as u64);
        Encoder {
            cfg,
            frames: 0,
            force_key: true,
            dec,
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

    /// Encodes one frame into one packet (a complete VP9 frame).
    pub fn encode(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        self.cfg.validate()?;
        if frame.width != self.cfg.width || frame.height != self.cfg.height {
            return Err(Error::invalid(format!(
                "frame is {}x{}, the encoder was configured for {}x{}",
                frame.width, frame.height, self.cfg.width, self.cfg.height
            )));
        }
        if frame.bit_depth != 8 || frame.chroma != ChromaFormat::Yuv420 {
            return Err(Error::unsupported(
                "the encoder writes profile 0 only: 8-bit 4:2:0",
            ));
        }
        let interval = self.cfg.keyframe_interval.max(1) as u64;
        let key = self.force_key || self.frames.is_multiple_of(interval);
        let h = self.header(key);
        let last: Option<Arc<RefFrame>> = if key { None } else { self.dec.ref_slot(0) };
        let key = key || last.is_none();
        let h = if key { self.header(true) } else { h };
        let comp = compressed_header(&h);
        let mut recon = None;
        let tiles = {
            let seg = Segmentation::default();
            let mut probs = Probs::default();
            let mut counts = Box::<Counts>::default();
            let refs: [Option<Arc<RefFrame>>; 3] = if key {
                Default::default()
            } else {
                [last.clone(), last.clone(), last.clone()]
            };
            let mut fd = FrameDec::new(&h, &seg, &mut probs, &mut counts, &[], None, refs);
            let src = tile::Source::new(frame, &h);
            let mut te = tile::TileEncoder::new(&self.cfg, &h, &src, last.as_deref());
            let t = te.encode_tiles(&mut fd);
            if cfg!(debug_assertions) && self.loop_filter_level() == 0 {
                recon = Some(fd.finish().0);
            }
            t
        };
        let mut w = BitWriter::default();
        self.uncompressed_header(&mut w, &h, comp.len())?;
        let mut packet = w.finish();
        packet.extend_from_slice(&comp);
        packet.extend_from_slice(&tiles);
        // Keep the references a decoder will have.
        let out = self.dec.decode(&packet).map_err(|e| {
            Error::invalid(format!(
                "internal error: the encoder's own frame does not decode: {e}"
            ))
        })?;
        // Debug builds check that the encoder reconstructed exactly what the
        // decoder decodes (without a loop filter the two are comparable).
        if let (Some(planes), Some(out)) = (recon, out) {
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
        self.frames += 1;
        self.force_key = false;
        Ok(packet)
    }

    fn loop_filter_level(&self) -> u8 {
        if let Some(l) = self.cfg.loop_filter_level {
            return l.min(63);
        }
        let q = self.cfg.quantizer as u32;
        if q == 0 {
            0
        } else {
            ((q * 10 + 32) / 64).min(63) as u8
        }
    }

    /// The header both the writer and the reconstruction use.
    fn header(&self, key: bool) -> FrameHeader {
        let (w, ht) = (self.cfg.width, self.cfg.height);
        let mi_cols = w.div_ceil(8);
        let mi_rows = ht.div_ceil(8);
        let sb64_cols = mi_cols.div_ceil(8);
        let sb64_rows = mi_rows.div_ceil(8);
        let mut min_log2 = 0;
        while (MAX_TILE_WIDTH_B64 << min_log2) < sb64_cols {
            min_log2 += 1;
        }
        let lossless = self.cfg.quantizer == 0;
        FrameHeader {
            profile: 0,
            frame_type: if key { KEY_FRAME } else { 1 },
            last_frame_type: KEY_FRAME,
            show_frame: true,
            error_resilient_mode: !key,
            bit_depth: 8,
            color_space: self.cfg.color_space,
            color_range: self.cfg.full_range,
            subsampling_x: 1,
            subsampling_y: 1,
            refresh_frame_flags: if key { 0xff } else { 0x01 },
            ref_frame_idx: [0, 1, 2],
            width: w,
            height: ht,
            render_width: w,
            render_height: ht,
            interpolation_filter: EIGHTTAP,
            frame_parallel_decoding_mode: true,
            base_q_idx: self.cfg.quantizer as i32,
            lossless,
            tile_cols_log2: min_log2,
            frame_is_intra: key,
            mi_cols,
            mi_rows,
            sb64_cols,
            sb64_rows,
            tx_mode: if lossless { ONLY_4X4 } else { ALLOW_32X32 },
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
        w.f(1, 0); // profile_low_bit
        w.f(1, 0); // profile_high_bit
        w.f(1, 0); // show_existing_frame
        w.f(1, h.frame_type as u32);
        w.f(1, 1); // show_frame
        w.f(1, h.error_resilient_mode as u32);
        if key {
            w.f(8, 0x49);
            w.f(8, 0x83);
            w.f(8, 0x42);
            // color_config(), profile 0
            w.f(3, h.color_space.bits());
            w.f(1, h.color_range as u32);
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
        w.f(6, self.loop_filter_level() as u32);
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

/// compressed_header(): the transform mode and no probability updates.
fn compressed_header(h: &FrameHeader) -> Vec<u8> {
    let mut e = BoolEncoder::new();
    let no = |e: &mut BoolEncoder| e.write(false, 252);
    if !h.lossless {
        e.literal(2, ALLOW_32X32 as u32);
        e.literal(1, 0); // tx_mode_select
    }
    // read_coef_probs(): no update for any transform size.
    let max_tx = TX_MODE_TO_BIGGEST_TX_SIZE[h.tx_mode as usize];
    for _ in 0..=max_tx {
        e.literal(1, 0);
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
