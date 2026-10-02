//! Prediction and reconstruction calls of the residual syntax: intra
//! prediction edges (8.5.1), inter prediction (8.5.2), dequantisation and
//! the inverse transform (8.6.2).

// Loops index arrays the way the specification's formulas do.
#![allow(clippy::needless_range_loop)]

use super::block::FrameDec;
use super::inter_scale;
use crate::consts::*;
use crate::dsp::{inter, intra, itx};
use crate::tables::*;
use crate::{Error, Result};

impl FrameDec<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict_intra(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        have_left: bool,
        have_above: bool,
        not_on_right: bool,
        tx_sz: u8,
        block_idx: usize,
    ) {
        let mode = if plane > 0 {
            self.b.uv_mode
        } else if self.b.mi_size >= BLOCK_8X8 {
            self.b.y_mode
        } else {
            self.b.sub_modes[block_idx]
        };
        let log2 = tx_sz as u32 + 2;
        let size = 1usize << log2;
        let (sx, sy) = if plane > 0 {
            (self.ss_x, self.ss_y)
        } else {
            (0, 0)
        };
        let max_x = ((self.mi_cols as usize * 8) >> sx) - 1;
        let max_y = ((self.mi_rows as usize * 8) >> sy) - 1;
        let base = 1i32 << (self.bit_depth - 1);
        let buf = &mut self.planes[plane];
        let stride = buf.stride;
        let mut above = [0i32; 65];
        let mut left = [0i32; 32];
        if have_above {
            let row = &buf.data[(y - 1) * stride..y * stride];
            for i in 0..size {
                above[1 + i] = row[max_x.min(x + i)] as i32;
            }
            if not_on_right && tx_sz == TX_4X4 {
                for i in size..2 * size {
                    above[1 + i] = row[max_x.min(x + i)] as i32;
                }
            } else {
                for i in size..2 * size {
                    above[1 + i] = above[size];
                }
            }
            above[0] = if have_left {
                row[max_x.min(x - 1)] as i32
            } else {
                base + 1
            };
        } else {
            for v in above.iter_mut().skip(1).take(2 * size) {
                *v = base - 1;
            }
            above[0] = base - 1;
        }
        if have_left {
            for (i, l) in left.iter_mut().enumerate().take(size) {
                *l = buf.data[max_y.min(y + i) * stride + x - 1] as i32;
            }
        } else {
            for l in left.iter_mut().take(size) {
                *l = base + 1;
            }
        }
        let off = y * stride + x;
        intra::predict(
            mode,
            log2,
            &above,
            &left,
            have_left,
            have_above,
            self.bit_depth,
            &mut buf.data[off..],
            stride,
        );
    }

    /// The inter prediction process (8.5.2) for one plane region.
    pub(crate) fn predict_inter(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        block_idx: usize,
    ) -> Result<()> {
        let is_compound = self.b.ref_frame[1] > INTRA_FRAME;
        let mut preds = [[0u16; 64 * 64]; 2];
        for ref_list in 0..1 + is_compound as usize {
            // Motion vector selection (8.5.2.1).
            let bm = &self.b.block_mvs[ref_list];
            let mv = if plane == 0 || self.b.mi_size >= BLOCK_8X8 {
                bm[block_idx]
            } else {
                match (self.ss_x, self.ss_y) {
                    (0, 0) => bm[block_idx],
                    (0, _) => [
                        q2(bm[block_idx][0] + bm[block_idx + 2][0]),
                        q2(bm[block_idx][1] + bm[block_idx + 2][1]),
                    ],
                    (_, 0) => [
                        q2(bm[block_idx][0] + bm[block_idx + 1][0]),
                        q2(bm[block_idx][1] + bm[block_idx + 1][1]),
                    ],
                    _ => [
                        q4(bm[0][0] + bm[1][0] + bm[2][0] + bm[3][0]),
                        q4(bm[0][1] + bm[1][1] + bm[2][1] + bm[3][1]),
                    ],
                }
            };
            // Motion vector clamping (8.5.2.2).
            let (sx, sy) = if plane > 0 {
                (self.ss_x as i32, self.ss_y as i32)
            } else {
                (0, 0)
            };
            let bh = NUM_8X8_HIGH[self.b.mi_size as usize] as i32;
            let bw = NUM_8X8_WIDE[self.b.mi_size as usize] as i32;
            let (mi_row, mi_col) = (self.b.mi_row as i32, self.b.mi_col as i32);
            let to_top = -((mi_row * 8) * 16) >> sy;
            let to_bottom = (((self.mi_rows as i32 - bh - mi_row) * 8) * 16) >> sy;
            let to_left = -((mi_col * 8) * 16) >> sx;
            let to_right = (((self.mi_cols as i32 - bw - mi_col) * 8) * 16) >> sx;
            let spel_left = (INTERP_EXTEND + ((bw * 8) >> sx)) << 4;
            let spel_right = spel_left - 16;
            let spel_top = (INTERP_EXTEND + ((bh * 8) >> sy)) << 4;
            let spel_bottom = spel_top - 16;
            let cmv = [
                ((2 * mv[0]) >> sy).clamp(to_top - spel_top, to_bottom + spel_bottom),
                ((2 * mv[1]) >> sx).clamp(to_left - spel_left, to_right + spel_right),
            ];
            // Motion vector scaling (8.5.2.3).
            let rf = self.b.ref_frame[ref_list];
            let r = self.refs[(rf - LAST_FRAME) as usize]
                .as_ref()
                .ok_or_else(|| Error::bitstream("missing reference"))?;
            let sc = inter_scale(r.width, r.height, self.h.width, self.h.height)
                .ok_or_else(|| Error::bitstream("reference frame scaled beyond 2:1 / 1:16"))?;
            let (start_x, start_y, step_x, step_y) =
                sc.position(plane > 0, self.ss_x, self.ss_y, x as i64, y as i64, cmv);
            let rp = &r.planes[plane];
            let last_x = ((r.width + sx as u32) >> sx) as i32 - 1;
            let last_y = ((r.height + sy as u32) >> sy) as i32 - 1;
            let refp = inter::RefPlane {
                data: &rp.data,
                stride: rp.stride,
                last_x,
                last_y,
            };
            inter::predict(
                &refp,
                start_x,
                start_y,
                step_x,
                step_y,
                w,
                h,
                self.b.interp_filter,
                self.bit_depth,
                &mut preds[ref_list][..w * h],
            );
        }
        let buf = &mut self.planes[plane];
        let stride = buf.stride;
        for i in 0..h {
            let row = &mut buf.data[(y + i) * stride + x..(y + i) * stride + x + w];
            if is_compound {
                for j in 0..w {
                    row[j] =
                        ((preds[0][i * w + j] as u32 + preds[1][i * w + j] as u32 + 1) >> 1) as u16;
                }
            } else {
                row.copy_from_slice(&preds[0][i * w..i * w + w]);
            }
        }
        Ok(())
    }

    fn qindex(&self) -> i32 {
        let seg = self.b.segment_id;
        if self.seg.active(seg, SEG_LVL_ALT_Q) {
            let mut data = self.seg.feature_data[seg as usize][SEG_LVL_ALT_Q] as i32;
            if !self.seg.abs_or_delta_update {
                data += self.h.base_q_idx;
            }
            data.clamp(0, 255)
        } else {
            self.h.base_q_idx
        }
    }

    /// The reconstruct process (8.6.2) for the coefficients in `self.coefs`.
    pub(crate) fn reconstruct(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tx_sz: u8,
        tx_type: u8,
        _eob: usize,
    ) {
        let bd_idx = ((self.bit_depth - 8) >> 1) as usize;
        let q = self.qindex();
        let (dc_delta, ac_delta) = if plane == 0 {
            (self.h.delta_q_y_dc, 0)
        } else {
            (self.h.delta_q_uv_dc, self.h.delta_q_uv_ac)
        };
        let dc_q = DC_QLOOKUP[bd_idx][(q + dc_delta).clamp(0, 255) as usize] as i64;
        let ac_q = AC_QLOOKUP[bd_idx][(q + ac_delta).clamp(0, 255) as usize] as i64;
        let dq_denom = if tx_sz == TX_32X32 { 2 } else { 1 };
        let n = 2 + tx_sz as u32;
        let n0 = 1usize << n;
        let block = &mut self.coefs[..n0 * n0];
        for (k, v) in block.iter_mut().enumerate() {
            if *v != 0 {
                let qq = if k == 0 { dc_q } else { ac_q };
                *v = ((*v as i64 * qq) / dq_denom).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            }
        }
        itx::inverse_transform_2d(block, n, tx_type, self.h.lossless);
        let max = (1i32 << self.bit_depth) - 1;
        let buf = &mut self.planes[plane];
        let stride = buf.stride;
        for i in 0..n0 {
            let row = &mut buf.data[(y + i) * stride + x..(y + i) * stride + x + n0];
            for j in 0..n0 {
                row[j] = (row[j] as i32)
                    .saturating_add(block[i * n0 + j])
                    .clamp(0, max) as u16;
            }
        }
        block.iter_mut().for_each(|v| *v = 0);
    }
}

#[inline]
fn q2(v: i32) -> i32 {
    (if v < 0 { v - 1 } else { v + 1 }) / 2
}

#[inline]
fn q4(v: i32) -> i32 {
    (if v < 0 { v - 2 } else { v + 2 }) / 4
}
