//! The uncompressed header (6.2) and the compressed header (6.3).

// Loops index arrays the way the specification's formulas do.
#![allow(clippy::needless_range_loop)]

use crate::bits::BitReader;
use crate::bool_coder::BoolDecoder;
use crate::consts::*;
use crate::frame::ColorSpace;
use crate::probs::Probs;
use crate::tables::*;
use crate::{Error, Result};

/// Loop filter parameters (6.2.8). The deltas persist from frame to frame.
#[derive(Clone, Debug)]
pub(crate) struct LoopFilter {
    pub level: u8,
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub ref_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
}

impl Default for LoopFilter {
    fn default() -> Self {
        LoopFilter {
            level: 0,
            sharpness: 0,
            delta_enabled: true,
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
        }
    }
}

/// Segmentation parameters (6.2.11). Feature data persists.
#[derive(Clone, Debug, Default)]
pub(crate) struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub tree_probs: [u8; 7],
    pub pred_probs: [u8; 3],
    pub temporal_update: bool,
    pub abs_or_delta_update: bool,
    pub feature_enabled: [[bool; 4]; MAX_SEGMENTS],
    pub feature_data: [[i16; 4]; MAX_SEGMENTS],
}

impl Segmentation {
    /// seg_feature_active( feature ) for `segment_id`.
    #[inline]
    pub(crate) fn active(&self, segment_id: u8, feature: usize) -> bool {
        self.enabled && self.feature_enabled[segment_id as usize][feature]
    }
}

/// State the uncompressed header reads and writes across frames.
pub(crate) struct Persistent {
    pub lf: LoopFilter,
    pub seg: Segmentation,
    /// RefFrameWidth / RefFrameHeight per slot (0 if never written).
    pub ref_sizes: [(u32, u32); 8],
    pub last_frame_type: u8,
    /// The colour configuration last read (inter frames inherit it).
    pub bit_depth: u32,
    pub subsampling: (u32, u32),
    pub color_space: ColorSpace,
    pub color_range: bool,
}

impl Default for Persistent {
    fn default() -> Self {
        Persistent {
            lf: LoopFilter::default(),
            seg: Segmentation::default(),
            ref_sizes: [(0, 0); 8],
            last_frame_type: KEY_FRAME,
            bit_depth: 8,
            subsampling: (1, 1),
            color_space: ColorSpace::Bt601,
            color_range: false,
        }
    }
}

/// Everything the uncompressed header says about one frame.
#[derive(Clone, Debug, Default)]
pub(crate) struct FrameHeader {
    pub profile: u8,
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: u8,
    pub frame_type: u8,
    /// LastFrameType: the frame_type of the previous frame.
    pub last_frame_type: u8,
    pub show_frame: bool,
    pub error_resilient_mode: bool,
    pub intra_only: bool,
    pub reset_frame_context: u8,
    pub bit_depth: u32,
    pub color_space: ColorSpace,
    pub color_range: bool,
    pub subsampling_x: u32,
    pub subsampling_y: u32,
    pub refresh_frame_flags: u8,
    pub ref_frame_idx: [u8; 3],
    /// Indexed by reference frame (INTRA_FRAME..ALTREF_FRAME).
    pub ref_frame_sign_bias: [bool; 4],
    pub width: u32,
    pub height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub allow_high_precision_mv: bool,
    pub interpolation_filter: u8,
    pub refresh_frame_context: bool,
    pub frame_parallel_decoding_mode: bool,
    pub frame_context_idx: u8,
    pub base_q_idx: i32,
    pub delta_q_y_dc: i32,
    pub delta_q_uv_dc: i32,
    pub delta_q_uv_ac: i32,
    pub lossless: bool,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    pub header_size_in_bytes: u32,
    /// Size of the uncompressed header in bytes (after trailing_bits).
    pub uncompressed_size: usize,
    /// Set when setup_past_independence was invoked.
    pub past_independence: bool,
    // Derived.
    pub frame_is_intra: bool,
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub sb64_cols: u32,
    pub sb64_rows: u32,
    // Compressed header.
    pub tx_mode: u8,
    pub reference_mode: u8,
    pub comp_fixed_ref: i8,
    pub comp_var_ref: [i8; 2],
}

impl FrameHeader {
    fn compute_image_size(&mut self) {
        self.mi_cols = (self.width + 7) >> 3;
        self.mi_rows = (self.height + 7) >> 3;
        self.sb64_cols = (self.mi_cols + 7) >> 3;
        self.sb64_rows = (self.mi_rows + 7) >> 3;
    }
}

fn frame_sync_code(r: &mut BitReader) -> Result<()> {
    let a = r.f(8)?;
    let b = r.f(8)?;
    let c = r.f(8)?;
    if (a, b, c) != (0x49, 0x83, 0x42) {
        return Err(Error::bitstream("invalid frame sync code"));
    }
    Ok(())
}

fn color_config(r: &mut BitReader, h: &mut FrameHeader) -> Result<()> {
    if h.profile >= 2 {
        h.bit_depth = if r.flag()? { 12 } else { 10 };
    } else {
        h.bit_depth = 8;
    }
    h.color_space = ColorSpace::from_bits(r.f(3)?);
    if h.color_space != ColorSpace::Rgb {
        h.color_range = r.flag()?;
        if h.profile == 1 || h.profile == 3 {
            h.subsampling_x = r.f(1)?;
            h.subsampling_y = r.f(1)?;
            if r.flag()? {
                return Err(Error::bitstream("reserved_zero is not zero"));
            }
            if h.subsampling_x == 1 && h.subsampling_y == 1 {
                return Err(Error::bitstream("4:2:0 is not allowed in profiles 1 and 3"));
            }
        } else {
            h.subsampling_x = 1;
            h.subsampling_y = 1;
        }
    } else {
        h.color_range = true;
        if h.profile == 1 || h.profile == 3 {
            h.subsampling_x = 0;
            h.subsampling_y = 0;
            if r.flag()? {
                return Err(Error::bitstream("reserved_zero is not zero"));
            }
        } else {
            return Err(Error::bitstream("sRGB is not allowed in profiles 0 and 2"));
        }
    }
    Ok(())
}

fn frame_size(r: &mut BitReader, h: &mut FrameHeader) -> Result<()> {
    h.width = r.f(16)? + 1;
    h.height = r.f(16)? + 1;
    h.compute_image_size();
    Ok(())
}

fn render_size(r: &mut BitReader, h: &mut FrameHeader) -> Result<()> {
    if r.flag()? {
        h.render_width = r.f(16)? + 1;
        h.render_height = r.f(16)? + 1;
    } else {
        h.render_width = h.width;
        h.render_height = h.height;
    }
    Ok(())
}

fn read_delta_q(r: &mut BitReader) -> Result<i32> {
    if r.flag()? { r.s(4) } else { Ok(0) }
}

fn read_prob(r: &mut BitReader) -> Result<u8> {
    if r.flag()? {
        Ok(r.f(8)? as u8)
    } else {
        Ok(255)
    }
}

/// What the caller must do after the uncompressed header asked for
/// setup_past_independence and save_probs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResetContexts {
    None,
    All,
    One(u8),
}

/// Parses uncompressed_header() and trailing_bits(). Updates `st` (loop
/// filter deltas, segmentation) as the syntax prescribes.
pub(crate) fn parse_uncompressed(
    data: &[u8],
    st: &mut Persistent,
) -> Result<(FrameHeader, ResetContexts)> {
    let mut r = BitReader::new(data);
    let mut h = FrameHeader::default();
    if r.f(2)? != 2 {
        return Err(Error::bitstream("frame_marker is not 2"));
    }
    let lo = r.f(1)?;
    let hi = r.f(1)?;
    h.profile = ((hi << 1) + lo) as u8;
    if h.profile == 3 && r.flag()? {
        return Err(Error::bitstream("reserved_zero is not zero"));
    }
    h.show_existing_frame = r.flag()?;
    if h.show_existing_frame {
        h.frame_to_show_map_idx = r.f(3)? as u8;
        h.header_size_in_bytes = 0;
        h.refresh_frame_flags = 0;
        r.byte_align();
        h.uncompressed_size = r.position() / 8;
        return Ok((h, ResetContexts::None));
    }
    let last_frame_type = st.last_frame_type;
    h.frame_type = r.f(1)? as u8;
    h.show_frame = r.flag()?;
    h.error_resilient_mode = r.flag()?;
    // LastFrameType is the previous frame's type; remember ours for the next.
    h.last_frame_type = last_frame_type;
    st.last_frame_type = h.frame_type;
    // Inter frames inherit the colour configuration.
    h.bit_depth = st.bit_depth;
    (h.subsampling_x, h.subsampling_y) = st.subsampling;
    h.color_space = st.color_space;
    h.color_range = st.color_range;
    if h.frame_type == KEY_FRAME {
        frame_sync_code(&mut r)?;
        color_config(&mut r, &mut h)?;
        frame_size(&mut r, &mut h)?;
        render_size(&mut r, &mut h)?;
        h.refresh_frame_flags = 0xff;
        h.frame_is_intra = true;
    } else {
        h.intra_only = if !h.show_frame { r.flag()? } else { false };
        h.frame_is_intra = h.intra_only;
        h.reset_frame_context = if !h.error_resilient_mode {
            r.f(2)? as u8
        } else {
            0
        };
        if h.intra_only {
            frame_sync_code(&mut r)?;
            if h.profile > 0 {
                color_config(&mut r, &mut h)?;
            } else {
                h.color_space = ColorSpace::Bt601;
                h.subsampling_x = 1;
                h.subsampling_y = 1;
                h.bit_depth = 8;
            }
            h.refresh_frame_flags = r.f(8)? as u8;
            frame_size(&mut r, &mut h)?;
            render_size(&mut r, &mut h)?;
        } else {
            h.refresh_frame_flags = r.f(8)? as u8;
            for i in 0..3 {
                h.ref_frame_idx[i] = r.f(3)? as u8;
                h.ref_frame_sign_bias[LAST_FRAME as usize + i] = r.flag()?;
            }
            // frame_size_with_refs()
            let mut found = false;
            for i in 0..3 {
                if r.flag()? {
                    let (w, ht) = st.ref_sizes[h.ref_frame_idx[i] as usize];
                    if w == 0 {
                        return Err(Error::bitstream(
                            "frame size taken from an empty reference slot",
                        ));
                    }
                    h.width = w;
                    h.height = ht;
                    found = true;
                    break;
                }
            }
            if !found {
                frame_size(&mut r, &mut h)?;
            } else {
                h.compute_image_size();
            }
            render_size(&mut r, &mut h)?;
            h.allow_high_precision_mv = r.flag()?;
            // read_interpolation_filter()
            h.interpolation_filter = if r.flag()? {
                SWITCHABLE
            } else {
                LITERAL_TO_TYPE[r.f(2)? as usize]
            };
        }
    }
    st.bit_depth = h.bit_depth;
    st.subsampling = (h.subsampling_x, h.subsampling_y);
    st.color_space = h.color_space;
    st.color_range = h.color_range;
    if !h.error_resilient_mode {
        h.refresh_frame_context = r.flag()?;
        h.frame_parallel_decoding_mode = r.flag()?;
    } else {
        h.refresh_frame_context = false;
        h.frame_parallel_decoding_mode = true;
    }
    h.frame_context_idx = r.f(2)? as u8;
    let mut reset = ResetContexts::None;
    if h.frame_is_intra || h.error_resilient_mode {
        // setup_past_independence()
        h.past_independence = true;
        st.seg.feature_data = [[0; 4]; MAX_SEGMENTS];
        st.seg.feature_enabled = [[false; 4]; MAX_SEGMENTS];
        st.seg.abs_or_delta_update = false;
        st.lf.delta_enabled = true;
        st.lf.ref_deltas = [1, 0, -1, -1];
        st.lf.mode_deltas = [0, 0];
        if h.frame_type == KEY_FRAME || h.error_resilient_mode || h.reset_frame_context == 3 {
            reset = ResetContexts::All;
        } else if h.reset_frame_context == 2 {
            reset = ResetContexts::One(h.frame_context_idx);
        }
        h.frame_context_idx = 0;
    }
    // loop_filter_params()
    st.lf.level = r.f(6)? as u8;
    st.lf.sharpness = r.f(3)? as u8;
    st.lf.delta_enabled = r.flag()?;
    if st.lf.delta_enabled && r.flag()? {
        for i in 0..4 {
            if r.flag()? {
                st.lf.ref_deltas[i] = r.s(6)? as i8;
            }
        }
        for i in 0..2 {
            if r.flag()? {
                st.lf.mode_deltas[i] = r.s(6)? as i8;
            }
        }
    }
    // quantization_params()
    h.base_q_idx = r.f(8)? as i32;
    h.delta_q_y_dc = read_delta_q(&mut r)?;
    h.delta_q_uv_dc = read_delta_q(&mut r)?;
    h.delta_q_uv_ac = read_delta_q(&mut r)?;
    h.lossless =
        h.base_q_idx == 0 && h.delta_q_y_dc == 0 && h.delta_q_uv_dc == 0 && h.delta_q_uv_ac == 0;
    // segmentation_params()
    let seg = &mut st.seg;
    seg.enabled = r.flag()?;
    seg.update_map = false;
    seg.temporal_update = false;
    if seg.enabled {
        seg.update_map = r.flag()?;
        if seg.update_map {
            for i in 0..7 {
                seg.tree_probs[i] = read_prob(&mut r)?;
            }
            seg.temporal_update = r.flag()?;
            for i in 0..3 {
                seg.pred_probs[i] = if seg.temporal_update {
                    read_prob(&mut r)?
                } else {
                    255
                };
            }
        }
        if r.flag()? {
            seg.abs_or_delta_update = r.flag()?;
            for i in 0..MAX_SEGMENTS {
                for j in 0..4 {
                    let mut v = 0i16;
                    let en = r.flag()?;
                    seg.feature_enabled[i][j] = en;
                    if en {
                        let bits = SEGMENTATION_FEATURE_BITS[j] as u32;
                        v = r.f(bits)? as i16;
                        if SEGMENTATION_FEATURE_SIGNED[j] == 1 && r.flag()? {
                            v = -v;
                        }
                    }
                    seg.feature_data[i][j] = v;
                }
            }
        }
    }
    // tile_info()
    let mut min_log2 = 0;
    while (MAX_TILE_WIDTH_B64 << min_log2) < h.sb64_cols {
        min_log2 += 1;
    }
    let mut max_log2 = 1;
    while (h.sb64_cols >> max_log2) >= MIN_TILE_WIDTH_B64 {
        max_log2 += 1;
    }
    max_log2 -= 1;
    h.tile_cols_log2 = min_log2;
    while h.tile_cols_log2 < max_log2 {
        if r.flag()? {
            h.tile_cols_log2 += 1;
        } else {
            break;
        }
    }
    h.tile_rows_log2 = r.f(1)?;
    if h.tile_rows_log2 == 1 {
        h.tile_rows_log2 += r.f(1)?;
    }
    h.header_size_in_bytes = r.f(16)?;
    r.byte_align();
    h.uncompressed_size = r.position() / 8;
    Ok((h, reset))
}

// ---------------------------------------------------------------------------
// Compressed header.

/// decode_term_subexp (6.3.4).
fn decode_term_subexp(d: &mut BoolDecoder) -> u32 {
    if d.literal(1) == 0 {
        return d.literal(4);
    }
    if d.literal(1) == 0 {
        return d.literal(4) + 16;
    }
    if d.literal(1) == 0 {
        return d.literal(5) + 32;
    }
    let v = d.literal(7);
    if v < 65 {
        return v + 64;
    }
    let bit = d.literal(1);
    (v << 1) - 1 + bit
}

fn inv_recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > 2 * m {
        v
    } else if v & 1 != 0 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

/// inv_remap_prob (6.3.5).
pub(crate) fn inv_remap_prob(delta: u32, prob: u8) -> u8 {
    let v = INV_MAP_TABLE[(delta as usize).min(254)] as i32;
    let m = prob as i32 - 1;
    let r = if (m << 1) <= 255 {
        1 + inv_recenter_nonneg(v, m)
    } else {
        255 - inv_recenter_nonneg(v, 255 - 1 - m)
    };
    r.clamp(1, 255) as u8
}

/// diff_update_prob (6.3.3).
fn diff_update_prob(d: &mut BoolDecoder, p: &mut u8) {
    if d.read(252) {
        let delta = decode_term_subexp(d);
        *p = inv_remap_prob(delta, *p);
    }
}

fn update_mv_prob(d: &mut BoolDecoder, p: &mut u8) {
    if d.read(252) {
        *p = ((d.literal(7) << 1) | 1) as u8;
    }
}

/// compressed_header() (6.3): fills tx_mode and the reference mode in `h`,
/// updates `probs`.
pub(crate) fn parse_compressed(data: &[u8], h: &mut FrameHeader, probs: &mut Probs) -> Result<()> {
    let mut d = BoolDecoder::new(data)?;
    // read_tx_mode()
    if h.lossless {
        h.tx_mode = ONLY_4X4;
    } else {
        h.tx_mode = d.literal(2) as u8;
        if h.tx_mode == ALLOW_32X32 {
            h.tx_mode += d.literal(1) as u8;
        }
    }
    if h.tx_mode == TX_MODE_SELECT {
        for i in 0..2 {
            diff_update_prob(&mut d, &mut probs.tx[1][i][0]);
        }
        for i in 0..2 {
            for j in 0..2 {
                diff_update_prob(&mut d, &mut probs.tx[2][i][j]);
            }
        }
        for i in 0..2 {
            for j in 0..3 {
                diff_update_prob(&mut d, &mut probs.tx[3][i][j]);
            }
        }
    }
    // read_coef_probs()
    let max_tx = TX_MODE_TO_BIGGEST_TX_SIZE[h.tx_mode as usize] as usize;
    for tx in 0..=max_tx {
        if d.literal(1) == 1 {
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..6 {
                        let max_l = if k == 0 { 3 } else { 6 };
                        for l in 0..max_l {
                            for m in 0..3 {
                                diff_update_prob(&mut d, &mut probs.coef[tx][i][j][k][l][m]);
                            }
                        }
                    }
                }
            }
        }
    }
    // read_skip_prob()
    for i in 0..3 {
        diff_update_prob(&mut d, &mut probs.skip[i]);
    }
    h.reference_mode = SINGLE_REFERENCE;
    if !h.frame_is_intra {
        for i in 0..7 {
            for j in 0..3 {
                diff_update_prob(&mut d, &mut probs.inter_mode[i][j]);
            }
        }
        if h.interpolation_filter == SWITCHABLE {
            for j in 0..4 {
                for i in 0..2 {
                    diff_update_prob(&mut d, &mut probs.interp_filter[j][i]);
                }
            }
        }
        for i in 0..4 {
            diff_update_prob(&mut d, &mut probs.is_inter[i]);
        }
        // frame_reference_mode()
        let sb = &h.ref_frame_sign_bias;
        let compound_allowed = (1..3).any(|i| sb[i + 1] != sb[1]);
        if compound_allowed {
            if d.literal(1) == 0 {
                h.reference_mode = SINGLE_REFERENCE;
            } else {
                h.reference_mode = if d.literal(1) == 0 {
                    COMPOUND_REFERENCE
                } else {
                    REFERENCE_MODE_SELECT
                };
                // setup_compound_reference_mode()
                let lf = LAST_FRAME as usize;
                let gf = GOLDEN_FRAME as usize;
                let af = ALTREF_FRAME as usize;
                if sb[lf] == sb[gf] {
                    h.comp_fixed_ref = ALTREF_FRAME;
                    h.comp_var_ref = [LAST_FRAME, GOLDEN_FRAME];
                } else if sb[lf] == sb[af] {
                    h.comp_fixed_ref = GOLDEN_FRAME;
                    h.comp_var_ref = [LAST_FRAME, ALTREF_FRAME];
                } else {
                    h.comp_fixed_ref = LAST_FRAME;
                    h.comp_var_ref = [GOLDEN_FRAME, ALTREF_FRAME];
                }
            }
        }
        // frame_reference_mode_probs()
        if h.reference_mode == REFERENCE_MODE_SELECT {
            for i in 0..5 {
                diff_update_prob(&mut d, &mut probs.comp_mode[i]);
            }
        }
        if h.reference_mode != COMPOUND_REFERENCE {
            for i in 0..5 {
                diff_update_prob(&mut d, &mut probs.single_ref[i][0]);
                diff_update_prob(&mut d, &mut probs.single_ref[i][1]);
            }
        }
        if h.reference_mode != SINGLE_REFERENCE {
            for i in 0..5 {
                diff_update_prob(&mut d, &mut probs.comp_ref[i]);
            }
        }
        // read_y_mode_probs()
        for i in 0..4 {
            for j in 0..9 {
                diff_update_prob(&mut d, &mut probs.y_mode[i][j]);
            }
        }
        // read_partition_probs()
        for i in 0..16 {
            for j in 0..3 {
                diff_update_prob(&mut d, &mut probs.partition[i][j]);
            }
        }
        // mv_probs()
        for j in 0..3 {
            update_mv_prob(&mut d, &mut probs.mv_joint[j]);
        }
        for i in 0..2 {
            update_mv_prob(&mut d, &mut probs.mv_sign[i]);
            for j in 0..10 {
                update_mv_prob(&mut d, &mut probs.mv_class[i][j]);
            }
            update_mv_prob(&mut d, &mut probs.mv_class0_bit[i]);
            for j in 0..10 {
                update_mv_prob(&mut d, &mut probs.mv_bits[i][j]);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    update_mv_prob(&mut d, &mut probs.mv_class0_fr[i][j][k]);
                }
            }
            for k in 0..3 {
                update_mv_prob(&mut d, &mut probs.mv_fr[i][k]);
            }
        }
        if h.allow_high_precision_mv {
            for i in 0..2 {
                update_mv_prob(&mut d, &mut probs.mv_class0_hp[i]);
                update_mv_prob(&mut d, &mut probs.mv_hp[i]);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_map_table_is_a_permutation_of_1_to_254() {
        let mut seen = [false; 256];
        for &v in INV_MAP_TABLE.iter().take(254) {
            assert!(!seen[v as usize], "{v} twice");
            seen[v as usize] = true;
        }
        assert!((1..=254).all(|v| seen[v]));
    }

    #[test]
    fn remap_zero_delta_moves_little() {
        // inv_map_table[0] = 7: delta 0 recentres to prob - 4 away from
        // the ends of the range.
        for p in 10..=245u8 {
            let q = inv_remap_prob(0, p);
            assert!((q as i32 - p as i32).abs() <= 4, "{p} -> {q}");
        }
    }
}
