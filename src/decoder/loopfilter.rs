//! The loop filter process (8.8): frame init, the superblock edge loop,
//! filter size and adaptive strength. The per-sample filters are in
//! `dsp::lf`.

// Loops index arrays the way the specification's formulas do.
#![allow(clippy::needless_range_loop)]

use super::{MiInfo, PlaneBuf};
use crate::consts::*;
use crate::dsp::lf;
use crate::header::{FrameHeader, LoopFilter, Segmentation};
use crate::tables::*;

/// LvlLookup[segment_id][ref][mode] (8.8.1).
fn lvl_lookup(lf: &LoopFilter, seg: &Segmentation) -> [[[u8; 2]; 4]; MAX_SEGMENTS] {
    let mut out = [[[0u8; 2]; 4]; MAX_SEGMENTS];
    let n_shift = lf.level >> 5;
    for (segment_id, o) in out.iter_mut().enumerate() {
        let mut lvl_seg = lf.level as i32;
        if seg.active(segment_id as u8, SEG_LVL_ALT_L) {
            let data = seg.feature_data[segment_id][SEG_LVL_ALT_L] as i32;
            lvl_seg = if seg.abs_or_delta_update {
                data
            } else {
                data + lf.level as i32
            };
            lvl_seg = lvl_seg.clamp(0, MAX_LOOP_FILTER);
        }
        if !lf.delta_enabled {
            for r in o.iter_mut() {
                *r = [lvl_seg as u8; 2];
            }
        } else {
            let intra = lvl_seg + ((lf.ref_deltas[INTRA_FRAME as usize] as i32) << n_shift);
            o[0] = [intra.clamp(0, MAX_LOOP_FILTER) as u8; 2];
            for rf in 1..4 {
                for mode in 0..2 {
                    let inter = lvl_seg
                        + ((lf.ref_deltas[rf] as i32) << n_shift)
                        + ((lf.mode_deltas[mode] as i32) << n_shift);
                    o[rf][mode] = inter.clamp(0, MAX_LOOP_FILTER) as u8;
                }
            }
        }
    }
    out
}

/// Applies the loop filter to the frame.
pub(crate) fn filter_frame(
    h: &FrameHeader,
    lf: &LoopFilter,
    seg: &Segmentation,
    mi: &[MiInfo],
    planes: &mut [PlaneBuf; 3],
) {
    let lvl = lvl_lookup(lf, seg);
    let shift = if lf.sharpness > 4 {
        2
    } else if lf.sharpness > 0 {
        1
    } else {
        0
    };
    // limit / blimit / thresh per level.
    let mut params = [(0i32, 0i32, 0i32); 64];
    for (l, p) in params.iter_mut().enumerate() {
        let l = l as i32;
        let limit = if lf.sharpness > 0 {
            (l >> shift).clamp(1, 9 - lf.sharpness as i32)
        } else {
            (l >> shift).max(1)
        };
        *p = (limit, 2 * (l + 2) + limit, l >> 4);
    }
    let mi_rows = h.mi_rows;
    let mi_cols = h.mi_cols;
    let mut row = 0;
    while row < mi_rows {
        let mut col = 0;
        while col < mi_cols {
            for (plane, buf) in planes.iter_mut().enumerate() {
                for pass in 0..2 {
                    superblock(h, mi, &lvl, &params, buf, plane, pass, row, col);
                }
            }
            col += 8;
        }
        row += 8;
    }
}

#[allow(clippy::too_many_arguments)]
fn superblock(
    h: &FrameHeader,
    mi: &[MiInfo],
    lvl: &[[[u8; 2]; 4]; MAX_SEGMENTS],
    params: &[(i32, i32, i32); 64],
    buf: &mut PlaneBuf,
    plane: usize,
    pass: usize,
    row: u32,
    col: u32,
) {
    let (sub_x, sub_y) = if plane > 0 {
        (h.subsampling_x, h.subsampling_y)
    } else {
        (0, 0)
    };
    let (sub, edge_len) = if pass == 0 {
        (sub_x, 64 >> sub_y)
    } else {
        (sub_y, 64 >> sub_x)
    };
    let mi_rows = h.mi_rows;
    let mi_cols = h.mi_cols;
    let stride = buf.stride;
    let step = if pass == 0 { 1 } else { stride };
    // Every decision below depends on the 8x8 block a sample's luma
    // position falls in, and a run of four samples along an edge never
    // leaves one (in any subsampling): decide once per run.
    for edge in 0..(16u32 >> sub) {
        for i in (0..edge_len).step_by(4) {
            let (x, y) = if pass == 0 {
                (col * 8 + edge * (4 << sub_x), row * 8 + (i << sub_y))
            } else {
                (col * 8 + (i << sub_x), row * 8 + edge * (4 << sub_y))
            };
            // onScreen (step 13).
            if x >= 8 * mi_cols
                || y >= 8 * mi_rows
                || (pass == 0 && x == 0)
                || (pass == 1 && y == 0)
            {
                continue;
            }
            let loop_col = ((x >> 3) >> sub_x) << sub_x;
            let loop_row = ((y >> 3) >> sub_y) << sub_y;
            let m = &mi[(loop_row * mi_cols + loop_col) as usize];
            let mi_size = m.mi_size;
            let tx_sz = if plane > 0 {
                if mi_size < BLOCK_8X8 {
                    TX_4X4
                } else {
                    let uv = SS_SIZE_LOOKUP[mi_size as usize][h.subsampling_x as usize]
                        [h.subsampling_y as usize];
                    m.tx_size.min(MAX_TXSIZE_LOOKUP[uv as usize])
                }
            } else {
                m.tx_size
            };
            let sb_size = if sub == 0 {
                mi_size
            } else {
                mi_size.max(BLOCK_16X16)
            };
            let is_intra = m.ref_frame[0] <= INTRA_FRAME;
            let is_block_edge = if pass == 0 {
                x % (8 * NUM_8X8_WIDE[sb_size as usize] as u32) == 0
            } else {
                y % (8 * NUM_8X8_HIGH[sb_size as usize] as u32) == 0
            };
            let is_tx_edge = if pass == 1
                && sub_x == 1
                && mi_cols & 1 == 1
                && edge & 1 == 1
                && (x + 8) >= mi_cols * 8
            {
                false
            } else {
                edge % (1 << tx_sz) == 0
            };
            let is_32_edge = edge % 8 == 0;
            let apply = is_block_edge || (is_tx_edge && is_intra) || (is_tx_edge && !m.skip);
            if !apply {
                continue;
            }
            // Filter size (8.8.3).
            let base_size = if tx_sz == TX_4X4 && is_32_edge {
                TX_8X8
            } else {
                tx_sz.min(TX_16X16)
            };
            let filter_size =
                if (pass == 0 && sub_x == 1 && base_size == TX_16X16 && (x >> 3) == mi_cols - 1)
                    || (pass == 1 && sub_y == 1 && base_size == TX_16X16 && (y >> 3) == mi_rows - 1)
                {
                    TX_8X8
                } else {
                    base_size
                };
            // Adaptive filter strength (8.8.4).
            let mode = m.y_mode;
            let mode_type = (mode == NEARESTMV || mode == NEARMV || mode == NEWMV) as usize;
            let rf = m.ref_frame[0].max(0) as usize;
            let l = lvl[m.segment_id as usize][rf][mode_type];
            if l == 0 {
                continue;
            }
            let (limit, blimit, thresh) = params[l as usize];
            let pos = (y >> sub_y) as usize * stride + (x >> sub_x) as usize;
            let along = if pass == 0 { stride } else { 1 };
            for k in 0..4 {
                lf::filter(
                    &mut buf.data,
                    pos + k * along,
                    step,
                    filter_size,
                    limit,
                    blimit,
                    thresh,
                    h.bit_depth,
                );
            }
        }
    }
}
