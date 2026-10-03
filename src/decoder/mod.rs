//! The decoder: frame-level process (section 8.1), reference slots, frame
//! contexts, output.

// Loops index arrays the way the specification's formulas do.
#![allow(clippy::needless_range_loop)]

pub(crate) mod block;
pub(crate) mod loopfilter;
mod mvpred;
mod recon;

use std::sync::Arc;

use crate::consts::*;
use crate::frame::{ChromaFormat, Frame};
use crate::header::{self, FrameHeader, Persistent, ResetContexts};
use crate::probs::{self, Counts, Probs};
use crate::superframe;
use crate::{Error, Result};

pub(crate) use block::{FrameDec, MiInfo};

/// The default limit of [`Decoder::set_max_pixels`]: 8192 x 8192.
pub const DEFAULT_MAX_PIXELS: u64 = 8192 * 8192;

/// One plane of samples, allocated to whole superblocks.
#[derive(Clone)]
pub(crate) struct PlaneBuf {
    pub data: Vec<u16>,
    pub stride: usize,
}

impl PlaneBuf {
    pub(crate) fn new(w: usize, h: usize) -> Self {
        PlaneBuf {
            data: vec![0; w * h],
            stride: w,
        }
    }
}

/// A decoded frame as kept in a reference slot (FrameStore and the
/// RefFrameWidth / RefSubsampling / RefBitDepth of section 8.10).
pub(crate) struct RefFrame {
    pub width: u32,
    pub height: u32,
    pub ss_x: u32,
    pub ss_y: u32,
    pub bit_depth: u32,
    pub planes: [PlaneBuf; 3],
    pub color_space: crate::frame::ColorSpace,
    pub color_range: bool,
    pub render: (u32, u32),
}

impl RefFrame {
    /// The visible frame, packed into a [`Frame`].
    fn to_frame(&self) -> Frame {
        let chroma = ChromaFormat::from_shifts(self.ss_x, self.ss_y);
        let mut f = Frame::new(self.width, self.height, self.bit_depth, chroma);
        f.color_space = self.color_space;
        f.full_range = self.color_range;
        f.render_width = self.render.0;
        f.render_height = self.render.1;
        let high = self.bit_depth > 8;
        for p in 0..3 {
            let pl = f.planes[p];
            let src = &self.planes[p];
            let (w, h) = (pl.width as usize, pl.height as usize);
            let dst = f.plane_mut(p);
            for y in 0..h {
                let row = &src.data[y * src.stride..y * src.stride + w];
                if high {
                    for (x, &v) in row.iter().enumerate() {
                        dst[2 * (y * w + x)..2 * (y * w + x) + 2].copy_from_slice(&v.to_le_bytes());
                    }
                } else {
                    for (x, &v) in row.iter().enumerate() {
                        dst[y * w + x] = v as u8;
                    }
                }
            }
        }
        f
    }
}

/// Motion vector and references of one 8x8 position, kept for the next
/// frame (PrevMvs / PrevRefFrames).
#[derive(Clone, Copy, Default)]
pub(crate) struct PrevMv {
    pub ref_frame: [i8; 2],
    pub mv: [[i32; 2]; 2],
}

/// A VP9 decoder.
///
/// Feed it one container packet at a time — a frame or a superframe — with
/// [`Decoder::decode`]. Decoding is single-threaded and bit-exact to the
/// specification's decoding process.
pub struct Decoder {
    st: Persistent,
    contexts: Box<[Probs; 4]>,
    refs: [Option<Arc<RefFrame>>; 8],
    prev_segment_ids: Vec<u8>,
    prev_mvs: Vec<PrevMv>,
    /// FrameWidth / FrameHeight the last time compute_image_size ran, and
    /// show_frame then.
    last_size: Option<(u32, u32)>,
    last_show_frame: bool,
    frames: u64,
    max_pixels: u64,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// A decoder with no reference frames: the stream must start with a key
    /// frame (or an intra-only frame).
    pub fn new() -> Self {
        Decoder {
            st: Persistent::default(),
            contexts: Box::new([
                Probs::default(),
                Probs::default(),
                Probs::default(),
                Probs::default(),
            ]),
            refs: Default::default(),
            prev_segment_ids: Vec::new(),
            prev_mvs: Vec::new(),
            last_size: None,
            last_show_frame: false,
            frames: 0,
            max_pixels: DEFAULT_MAX_PIXELS,
        }
    }

    /// Refuses frames of more than `pixels` luma samples (width x height)
    /// with [`Error::Unsupported`] instead of allocating for them. The
    /// default is [`DEFAULT_MAX_PIXELS`]; VP9 allows up to 65536 x 65536.
    pub fn set_max_pixels(&mut self, pixels: u64) {
        self.max_pixels = pixels;
    }

    /// Decodes one packet (a frame, or a superframe of several). Returns the
    /// frame to show, if the packet shows one; if a superframe shows more
    /// than one, the last is returned ([`Decoder::decode_all`] returns all).
    pub fn decode(&mut self, packet: &[u8]) -> Result<Option<Frame>> {
        Ok(self.decode_all(packet)?.pop())
    }

    /// Decodes one packet and returns every frame it shows, in order.
    pub fn decode_all(&mut self, packet: &[u8]) -> Result<Vec<Frame>> {
        let mut out = Vec::new();
        for f in superframe::split(packet) {
            if let Some(frame) = self.decode_frame(f)? {
                out.push(frame);
            }
        }
        Ok(out)
    }

    /// The frame in reference slot `i`.
    pub(crate) fn ref_slot(&self, i: usize) -> Option<Arc<RefFrame>> {
        self.refs[i].clone()
    }

    /// Number of frames decoded so far (shown or not).
    pub fn frames_decoded(&self) -> u64 {
        self.frames
    }

    fn decode_frame(&mut self, data: &[u8]) -> Result<Option<Frame>> {
        if data.is_empty() {
            return Err(Error::bitstream("empty frame"));
        }
        let saved_last_type = self.st.last_frame_type;
        let (mut h, reset) = header::parse_uncompressed(data, &mut self.st)?;
        if h.show_existing_frame {
            self.st.last_frame_type = saved_last_type;
            let r = self.refs[h.frame_to_show_map_idx as usize]
                .as_ref()
                .ok_or_else(|| Error::bitstream("show_existing_frame of an empty slot"))?;
            self.frames += 1;
            return Ok(Some(r.to_frame()));
        }
        if h.width as u64 * h.height as u64 > self.max_pixels {
            return Err(Error::unsupported(format!(
                "{}x{} is above the decoder's frame size limit",
                h.width, h.height
            )));
        }
        if h.header_size_in_bytes == 0 {
            return Err(Error::bitstream("compressed header of size 0"));
        }
        if h.profile >= 4 {
            return Err(Error::unsupported("profile above 3"));
        }
        // The frame's references (inter frames).
        let mut active_refs: [Option<Arc<RefFrame>>; 3] = Default::default();
        if !h.frame_is_intra {
            for i in 0..3 {
                let r = self.refs[h.ref_frame_idx[i] as usize]
                    .clone()
                    .ok_or_else(|| Error::bitstream("reference to an empty slot"))?;
                if r.ss_x != h.subsampling_x
                    || r.ss_y != h.subsampling_y
                    || r.bit_depth != h.bit_depth
                {
                    return Err(Error::bitstream("reference frame of another format"));
                }
                active_refs[i] = Some(r);
            }
            // Bit depth and subsampling of an inter frame are those of its
            // references (the last colour config read).
        }
        // compute_image_size() semantics (7.2.6).
        let size = (h.width, h.height);
        let first = self.last_size.is_none();
        let same_size = self.last_size == Some(size);
        let mi_len = (h.mi_rows * h.mi_cols) as usize;
        let use_prev_frame_mvs = !first
            && same_size
            && self.last_show_frame
            && !h.error_resilient_mode
            && !h.frame_is_intra;
        if !same_size {
            self.prev_segment_ids = vec![0; mi_len];
        }
        self.last_size = Some(size);
        self.last_show_frame = h.show_frame;
        if h.past_independence {
            self.prev_segment_ids.iter_mut().for_each(|v| *v = 0);
        }
        // Probability contexts: setup_past_independence + save_probs.
        match reset {
            ResetContexts::All => {
                for c in self.contexts.iter_mut() {
                    *c = Probs::default();
                }
            }
            ResetContexts::One(i) => self.contexts[i as usize] = Probs::default(),
            ResetContexts::None => {}
        }
        // load_probs( frame_context_idx ) and load_probs2( ... ).
        let mut probs = self.contexts[h.frame_context_idx as usize].clone();
        let comp_start = h.uncompressed_size;
        let comp_end = comp_start + h.header_size_in_bytes as usize;
        if comp_end > data.len() {
            return Err(Error::bitstream(
                "compressed header runs past the end of the frame",
            ));
        }
        header::parse_compressed(&data[comp_start..comp_end], &mut h, &mut probs)?;
        let mut counts = Box::<Counts>::default();
        let seg = self.st.seg.clone();
        let lf = self.st.lf.clone();
        let mut fd = block::FrameDec::new(
            &h,
            &seg,
            &mut probs,
            &mut counts,
            &self.prev_segment_ids,
            if use_prev_frame_mvs {
                Some(&self.prev_mvs[..])
            } else {
                None
            },
            active_refs,
        );
        fd.decode_tiles(&data[comp_end..])?;
        let (planes, mi) = fd.finish();
        let mut planes = planes;
        // Loop filter (8.8).
        if lf.level != 0 {
            loopfilter::filter_frame(&h, &lf, &seg, &mi, &mut planes);
        }
        // Backward adaptation (refresh_probs, 6.1.2).
        if !h.error_resilient_mode && !h.frame_parallel_decoding_mode {
            let pre = &self.contexts[h.frame_context_idx as usize];
            let mut adapted = probs.clone();
            adapted.load_except_tx_skip(pre);
            probs::adapt_coef_probs(
                &mut adapted,
                &counts,
                h.frame_is_intra,
                h.last_frame_type == KEY_FRAME,
            );
            if !h.frame_is_intra {
                adapted.load_tx_skip(pre);
                probs::adapt_noncoef_probs(
                    &mut adapted,
                    &counts,
                    h.interpolation_filter == SWITCHABLE,
                    h.tx_mode == TX_MODE_SELECT,
                    h.allow_high_precision_mv,
                );
            }
            probs = adapted;
        }
        if h.refresh_frame_context {
            self.contexts[h.frame_context_idx as usize] = probs;
        }
        // Segmentation map for the next frame (8.1 step 3).
        if seg.enabled && seg.update_map {
            self.prev_segment_ids.clear();
            self.prev_segment_ids
                .extend(mi.iter().map(|m| m.segment_id));
        }
        // PrevMvs / PrevRefFrames (8.10 step 2).
        self.prev_mvs.clear();
        self.prev_mvs.extend(mi.iter().map(|m| PrevMv {
            ref_frame: m.ref_frame,
            mv: [m.mv[0][3], m.mv[1][3]],
        }));
        // Reference update (8.10 step 1).
        let cur = Arc::new(RefFrame {
            width: h.width,
            height: h.height,
            ss_x: h.subsampling_x,
            ss_y: h.subsampling_y,
            bit_depth: h.bit_depth,
            planes,
            color_space: h.color_space,
            color_range: h.color_range,
            render: (h.render_width, h.render_height),
        });
        for i in 0..8 {
            if (h.refresh_frame_flags >> i) & 1 == 1 {
                self.refs[i] = Some(cur.clone());
                self.st.ref_sizes[i] = (h.width, h.height);
            }
        }
        self.frames += 1;
        Ok(if h.show_frame {
            Some(cur.to_frame())
        } else {
            None
        })
    }
}

/// Exposed for the encoder's tests: the header of a frame.
#[allow(dead_code)]
pub(crate) fn peek_header(data: &[u8]) -> Result<FrameHeader> {
    let mut st = Persistent::default();
    Ok(header::parse_uncompressed(data, &mut st)?.0)
}

/// The reference scale factors of 8.5.2.3, in units of 1 / (1 << 14).
pub(crate) struct Scale {
    x_scale: i64,
    y_scale: i64,
}

/// xScale and yScale for a reference of `ref_w` x `ref_h` predicting a
/// frame of `w` x `h`; `None` beyond the ratios the specification allows.
pub(crate) fn inter_scale(ref_w: u32, ref_h: u32, w: u32, h: u32) -> Option<Scale> {
    if 2 * w < ref_w || 2 * h < ref_h || w > 16 * ref_w || h > 16 * ref_h {
        return None;
    }
    Some(Scale {
        x_scale: ((ref_w as i64) << 14) / w as i64,
        y_scale: ((ref_h as i64) << 14) / h as i64,
    })
}

impl Scale {
    /// startX, startY, stepX, stepY for the region at (`x`, `y`) of a plane
    /// with the clamped motion vector `cmv`.
    pub(crate) fn position(
        &self,
        chroma: bool,
        ss_x: u32,
        ss_y: u32,
        x: i64,
        y: i64,
        cmv: [i32; 2],
    ) -> (i32, i32, i32, i32) {
        let base_x = (x * self.x_scale) >> 14;
        let base_y = (y * self.y_scale) >> 14;
        let luma_x = if chroma { x << ss_x } else { x };
        let luma_y = if chroma { y << ss_y } else { y };
        let frac_x = ((16 * luma_x * self.x_scale) >> 14) & 15;
        let frac_y = ((16 * luma_y * self.y_scale) >> 14) & 15;
        let dx = ((cmv[1] as i64 * self.x_scale) >> 14) + frac_x;
        let dy = ((cmv[0] as i64 * self.y_scale) >> 14) + frac_y;
        let step_x = (16 * self.x_scale) >> 14;
        let step_y = (16 * self.y_scale) >> 14;
        (
            ((base_x << 4) + dx) as i32,
            ((base_y << 4) + dy) as i32,
            step_x as i32,
            step_y as i32,
        )
    }
}
