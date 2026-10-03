//! Tile, partition and block syntax (6.4): mode info, residual tokens, and
//! the calls into prediction and reconstruction.

use std::sync::Arc;

use super::{PlaneBuf, PrevMv, RefFrame};
use crate::bool_coder::BoolDecoder;
use crate::consts::*;
use crate::header::{FrameHeader, Segmentation};
use crate::probs::{Counts, Probs};
use crate::tables::*;
use crate::{Error, Result};

pub(crate) type Mv = [i32; 2];

/// What the frame remembers about each 8x8 position (the Skips, TxSizes,
/// MiSizes, YModes, SegmentIds, RefFrames, InterpFilters, Mvs, SubMvs and
/// SubModes arrays of 6.4.4).
#[derive(Clone, Copy)]
pub(crate) struct MiInfo {
    pub mi_size: u8,
    pub skip: bool,
    pub tx_size: u8,
    pub y_mode: u8,
    pub sub_modes: [u8; 4],
    pub segment_id: u8,
    pub ref_frame: [i8; 2],
    pub interp_filter: u8,
    /// SubMvs[list][b]; Mvs[list] is `mv[list][3]`.
    pub mv: [[Mv; 4]; 2],
}

impl Default for MiInfo {
    fn default() -> Self {
        MiInfo {
            mi_size: 0,
            skip: false,
            tx_size: 0,
            y_mode: 0,
            sub_modes: [0; 4],
            segment_id: 0,
            ref_frame: [INTRA_FRAME, NONE],
            interp_filter: 0,
            mv: [[[0; 2]; 4]; 2],
        }
    }
}

/// The state of the block being decoded.
#[derive(Default, Clone)]
pub(crate) struct Block {
    pub mi_row: u32,
    pub mi_col: u32,
    pub mi_size: u8,
    pub avail_u: bool,
    pub avail_l: bool,
    pub segment_id: u8,
    pub skip: bool,
    pub tx_size: u8,
    pub is_inter: bool,
    pub y_mode: u8,
    pub sub_modes: [u8; 4],
    pub uv_mode: u8,
    pub ref_frame: [i8; 2],
    pub interp_filter: u8,
    pub block_mvs: [[Mv; 4]; 2],
    // Neighbours (copies of the MiInfo above and to the left).
    pub above: Option<MiInfo>,
    pub left: Option<MiInfo>,
    // Motion vector prediction outputs.
    pub mode_context: [u8; 4],
    pub nearest_mv: [Mv; 2],
    pub near_mv: [Mv; 2],
    pub best_mv: [Mv; 2],
    pub mv: [Mv; 2],
}

pub(crate) struct FrameDec<'a> {
    pub h: &'a FrameHeader,
    pub seg: &'a Segmentation,
    pub probs: &'a mut Probs,
    pub counts: &'a mut Counts,
    pub prev_segment_ids: &'a [u8],
    pub prev_mvs: Option<&'a [PrevMv]>,
    /// The LAST, GOLDEN and ALTREF frames.
    pub refs: [Option<Arc<RefFrame>>; 3],
    pub planes: [PlaneBuf; 3],
    pub mi: Vec<MiInfo>,
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub ss_x: u32,
    pub ss_y: u32,
    pub bit_depth: u32,
    // Contexts.
    pub above_nonzero: [Vec<u8>; 3],
    pub left_nonzero: [Vec<u8>; 3],
    pub above_partition: Vec<u8>,
    pub left_partition: Vec<u8>,
    pub above_seg_pred: Vec<u8>,
    pub left_seg_pred: Vec<u8>,
    // Tile.
    pub mi_row_start: u32,
    pub mi_row_end: u32,
    pub mi_col_start: u32,
    pub mi_col_end: u32,
    pub b: Block,
    pub eob_total: u32,
    pub coefs: Vec<i32>,
    pub token_cache: Vec<u8>,
    /// Whether every tile decoded so far ended with zero padding (9.2.3).
    pub padding_ok: bool,
}

/// Probability of node `node` of the token tree (pareto, 9.3.2).
#[inline]
pub(crate) fn pareto(node: usize, prob: u8) -> u8 {
    if node < 2 {
        return prob;
    }
    let x = ((prob as usize).max(1) - 1) / 2;
    if prob & 1 != 0 {
        PARETO_TABLE[x][node - 2]
    } else {
        ((PARETO_TABLE[x][node - 2] as u32 + PARETO_TABLE[(x + 1).min(127)][node - 2] as u32) >> 1)
            as u8
    }
}

impl<'a> FrameDec<'a> {
    pub(crate) fn new(
        h: &'a FrameHeader,
        seg: &'a Segmentation,
        probs: &'a mut Probs,
        counts: &'a mut Counts,
        prev_segment_ids: &'a [u8],
        prev_mvs: Option<&'a [PrevMv]>,
        refs: [Option<Arc<RefFrame>>; 3],
    ) -> Self {
        let w = (h.sb64_cols * 64) as usize;
        let ht = (h.sb64_rows * 64) as usize;
        let (sx, sy) = (h.subsampling_x as usize, h.subsampling_y as usize);
        let planes = [
            PlaneBuf::new(w, ht),
            PlaneBuf::new(w >> sx, ht >> sy),
            PlaneBuf::new(w >> sx, ht >> sy),
        ];
        let n4w = (h.sb64_cols * 16) as usize + 16;
        let n4h = (h.sb64_rows * 16) as usize + 16;
        FrameDec {
            h,
            seg,
            probs,
            counts,
            prev_segment_ids,
            prev_mvs,
            refs,
            planes,
            mi: vec![MiInfo::default(); (h.mi_rows * h.mi_cols) as usize],
            mi_cols: h.mi_cols,
            mi_rows: h.mi_rows,
            ss_x: h.subsampling_x,
            ss_y: h.subsampling_y,
            bit_depth: h.bit_depth,
            above_nonzero: [vec![0; n4w], vec![0; n4w], vec![0; n4w]],
            left_nonzero: [vec![0; n4h], vec![0; n4h], vec![0; n4h]],
            above_partition: vec![0; (h.sb64_cols * 8) as usize + 8],
            left_partition: vec![0; (h.sb64_rows * 8) as usize + 8],
            above_seg_pred: vec![0; (h.sb64_cols * 8) as usize + 8],
            left_seg_pred: vec![0; (h.sb64_rows * 8) as usize + 8],
            mi_row_start: 0,
            mi_row_end: 0,
            mi_col_start: 0,
            mi_col_end: 0,
            b: Block::default(),
            eob_total: 0,
            coefs: vec![0; 1024],
            token_cache: vec![0; 1024],
            padding_ok: true,
        }
    }

    pub(crate) fn finish(self) -> ([PlaneBuf; 3], Vec<MiInfo>) {
        (self.planes, self.mi)
    }

    /// decode_tiles( sz ) (6.4).
    pub(crate) fn decode_tiles(&mut self, data: &[u8]) -> Result<()> {
        let tile_cols = 1u32 << self.h.tile_cols_log2;
        let tile_rows = 1u32 << self.h.tile_rows_log2;
        // clear_above_context()
        for p in self.above_nonzero.iter_mut() {
            p.iter_mut().for_each(|v| *v = 0);
        }
        self.above_partition.iter_mut().for_each(|v| *v = 0);
        self.above_seg_pred.iter_mut().for_each(|v| *v = 0);
        let mut pos = 0usize;
        for tile_row in 0..tile_rows {
            for tile_col in 0..tile_cols {
                let last = tile_row == tile_rows - 1 && tile_col == tile_cols - 1;
                let size = if last {
                    data.len() - pos
                } else {
                    if pos + 4 > data.len() {
                        return Err(Error::bitstream("tile size runs past the end of the frame"));
                    }
                    let s = u32::from_be_bytes([
                        data[pos],
                        data[pos + 1],
                        data[pos + 2],
                        data[pos + 3],
                    ]) as usize;
                    pos += 4;
                    s
                };
                if pos + size > data.len() {
                    return Err(Error::bitstream("tile runs past the end of the frame"));
                }
                let tile = &data[pos..pos + size];
                pos += size;
                self.mi_row_start = tile_offset(tile_row, self.mi_rows, self.h.tile_rows_log2);
                self.mi_row_end = tile_offset(tile_row + 1, self.mi_rows, self.h.tile_rows_log2);
                self.mi_col_start = tile_offset(tile_col, self.mi_cols, self.h.tile_cols_log2);
                self.mi_col_end = tile_offset(tile_col + 1, self.mi_cols, self.h.tile_cols_log2);
                let mut d = BoolDecoder::new(tile)?;
                self.decode_tile(&mut d)?;
                self.padding_ok &= d.padding_is_zero();
            }
        }
        Ok(())
    }

    fn decode_tile(&mut self, d: &mut BoolDecoder) -> Result<()> {
        let mut r = self.mi_row_start;
        while r < self.mi_row_end {
            // clear_left_context()
            for p in self.left_nonzero.iter_mut() {
                p.iter_mut().for_each(|v| *v = 0);
            }
            self.left_partition.iter_mut().for_each(|v| *v = 0);
            self.left_seg_pred.iter_mut().for_each(|v| *v = 0);
            let mut c = self.mi_col_start;
            while c < self.mi_col_end {
                self.decode_partition(d, r, c, BLOCK_64X64)?;
                c += 8;
            }
            r += 8;
        }
        Ok(())
    }

    fn decode_partition(&mut self, d: &mut BoolDecoder, r: u32, c: u32, bsize: u8) -> Result<()> {
        if r >= self.mi_rows || c >= self.mi_cols {
            return Ok(());
        }
        let num8x8 = NUM_8X8_WIDE[bsize as usize] as u32;
        let half = num8x8 >> 1;
        let has_rows = (r + half) < self.mi_rows;
        let has_cols = (c + half) < self.mi_cols;
        // Partition context.
        let bsl = MI_WIDTH_LOG2[bsize as usize] as u32;
        let boffset = MI_WIDTH_LOG2[BLOCK_64X64 as usize] as u32 - bsl;
        let mut above = 0u8;
        let mut left = 0u8;
        for i in 0..num8x8 {
            above |= self.above_partition[(c + i) as usize];
            left |= self.left_partition[(r + i) as usize];
        }
        let above = (above & (1 << boffset)) != 0;
        let left = (left & (1 << boffset)) != 0;
        let ctx = (bsl * 4 + left as u32 * 2 + above as u32) as usize;
        let probs = if self.h.frame_is_intra {
            KF_PARTITION_PROBS[ctx]
        } else {
            self.probs.partition[ctx]
        };
        let partition = if has_rows && has_cols {
            d.tree(&PARTITION_TREE, |n| probs[n])
        } else if has_cols {
            if d.read(probs[1]) {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows {
            if d.read(probs[2]) {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        };
        if !self.h.frame_is_intra {
            self.counts.partition[ctx][partition as usize] += 1;
        }
        let subsize = SUBSIZE_LOOKUP[partition as usize][bsize as usize];
        if subsize == BLOCK_INVALID {
            return Err(Error::bitstream("invalid partition"));
        }
        if subsize < BLOCK_8X8 || partition == PARTITION_NONE {
            self.decode_block(d, r, c, subsize)?;
        } else if partition == PARTITION_HORZ {
            self.decode_block(d, r, c, subsize)?;
            if has_rows {
                self.decode_block(d, r + half, c, subsize)?;
            }
        } else if partition == PARTITION_VERT {
            self.decode_block(d, r, c, subsize)?;
            if has_cols {
                self.decode_block(d, r, c + half, subsize)?;
            }
        } else {
            self.decode_partition(d, r, c, subsize)?;
            self.decode_partition(d, r, c + half, subsize)?;
            self.decode_partition(d, r + half, c, subsize)?;
            self.decode_partition(d, r + half, c + half, subsize)?;
        }
        if bsize == BLOCK_8X8 || partition != PARTITION_SPLIT {
            let a = 15 >> B_WIDTH_LOG2[subsize as usize];
            let l = 15 >> B_HEIGHT_LOG2[subsize as usize];
            for i in 0..num8x8 {
                self.above_partition[(c + i) as usize] = a;
                self.left_partition[(r + i) as usize] = l;
            }
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn mi_at(&self, r: u32, c: u32) -> &MiInfo {
        &self.mi[(r * self.mi_cols + c) as usize]
    }

    fn decode_block(&mut self, d: &mut BoolDecoder, r: u32, c: u32, subsize: u8) -> Result<()> {
        if self.ss_x + self.ss_y > 0 && subsize >= BLOCK_8X8 {
            let uv = SS_SIZE_LOOKUP[subsize as usize][self.ss_x as usize][self.ss_y as usize];
            if uv == BLOCK_INVALID {
                return Err(Error::bitstream(
                    "block size invalid for the chroma subsampling",
                ));
            }
        }
        let avail_u = r > 0;
        let avail_l = c > self.mi_col_start;
        self.b = Block {
            mi_row: r,
            mi_col: c,
            mi_size: subsize,
            avail_u,
            avail_l,
            above: if avail_u {
                Some(*self.mi_at(r - 1, c))
            } else {
                None
            },
            left: if avail_l {
                Some(*self.mi_at(r, c - 1))
            } else {
                None
            },
            ..Block::default()
        };
        if self.h.frame_is_intra {
            self.intra_frame_mode_info(d);
        } else {
            self.inter_frame_mode_info(d)?;
        }
        self.eob_total = 0;
        self.residual(d)?;
        let b = &mut self.b;
        if b.is_inter && subsize >= BLOCK_8X8 && self.eob_total == 0 {
            b.skip = true;
        }
        let info = MiInfo {
            mi_size: subsize,
            skip: b.skip,
            tx_size: b.tx_size,
            y_mode: b.y_mode,
            sub_modes: b.sub_modes,
            segment_id: b.segment_id,
            ref_frame: b.ref_frame,
            interp_filter: b.interp_filter,
            mv: b.block_mvs,
        };
        let bh = NUM_8X8_HIGH[subsize as usize] as u32;
        let bw = NUM_8X8_WIDE[subsize as usize] as u32;
        for y in r..(r + bh).min(self.mi_rows) {
            let row = (y * self.mi_cols) as usize;
            for x in c..(c + bw).min(self.mi_cols) {
                self.mi[row + x as usize] = info;
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // Mode info (6.4.5 - 6.4.17).

    fn intra_frame_mode_info(&mut self, d: &mut BoolDecoder) {
        // intra_segment_id()
        self.b.segment_id = if self.seg.enabled && self.seg.update_map {
            d.tree(&SEGMENT_TREE, |n| self.seg.tree_probs[n])
        } else {
            0
        };
        self.read_skip(d);
        self.read_tx_size(d, true);
        self.b.ref_frame = [INTRA_FRAME, NONE];
        self.b.is_inter = false;
        let above = self.b.above;
        let left = self.b.left;
        if self.b.mi_size >= BLOCK_8X8 {
            let am = above.map_or(DC_PRED, |m| m.sub_modes[2]);
            let lm = left.map_or(DC_PRED, |m| m.sub_modes[1]);
            let p = &KF_Y_MODE_PROBS[am as usize][lm as usize];
            let mode = d.tree(&INTRA_MODE_TREE, |n| p[n]);
            self.b.y_mode = mode;
            self.b.sub_modes = [mode; 4];
        } else {
            let n4w = NUM_4X4_WIDE[self.b.mi_size as usize] as usize;
            let n4h = NUM_4X4_HIGH[self.b.mi_size as usize] as usize;
            let mut mode = DC_PRED;
            let mut idy = 0;
            while idy < 2 {
                let mut idx = 0;
                while idx < 2 {
                    let am = if idy > 0 {
                        self.b.sub_modes[idx]
                    } else {
                        above.map_or(DC_PRED, |m| m.sub_modes[2 + idx])
                    };
                    let lm = if idx > 0 {
                        self.b.sub_modes[idy * 2]
                    } else {
                        left.map_or(DC_PRED, |m| m.sub_modes[1 + idy * 2])
                    };
                    let p = &KF_Y_MODE_PROBS[am as usize][lm as usize];
                    mode = d.tree(&INTRA_MODE_TREE, |n| p[n]);
                    for y2 in 0..n4h {
                        for x2 in 0..n4w {
                            self.b.sub_modes[(idy + y2) * 2 + idx + x2] = mode;
                        }
                    }
                    idx += n4w;
                }
                idy += n4h;
            }
            self.b.y_mode = mode;
        }
        let p = &KF_UV_MODE_PROBS[self.b.y_mode as usize];
        self.b.uv_mode = d.tree(&INTRA_MODE_TREE, |n| p[n]);
    }

    fn read_skip(&mut self, d: &mut BoolDecoder) {
        if self.seg.active(self.b.segment_id, SEG_LVL_SKIP) {
            self.b.skip = true;
        } else {
            let ctx = self.b.above.map_or(0, |m| m.skip as usize)
                + self.b.left.map_or(0, |m| m.skip as usize);
            self.b.skip = d.read(self.probs.skip[ctx]);
            self.counts.skip[ctx][self.b.skip as usize] += 1;
        }
    }

    fn read_tx_size(&mut self, d: &mut BoolDecoder, allow_select: bool) {
        let max_tx = MAX_TXSIZE_LOOKUP[self.b.mi_size as usize];
        if allow_select && self.h.tx_mode == TX_MODE_SELECT && self.b.mi_size >= BLOCK_8X8 {
            let mut above = max_tx;
            let mut left = max_tx;
            if let Some(m) = self.b.above
                && !m.skip
            {
                above = m.tx_size;
            }
            if let Some(m) = self.b.left
                && !m.skip
            {
                left = m.tx_size;
            }
            if !self.b.avail_l {
                left = above;
            }
            if !self.b.avail_u {
                above = left;
            }
            let ctx = ((above + left) > max_tx) as usize;
            let p = self.probs.tx[max_tx as usize][ctx];
            let tree: &[i8] = match max_tx {
                TX_32X32 => &TX_SIZE_32_TREE,
                TX_16X16 => &TX_SIZE_16_TREE,
                _ => &TX_SIZE_8_TREE,
            };
            let v = d.tree(tree, |n| p[n]);
            self.counts.tx[max_tx as usize][ctx][v as usize] += 1;
            self.b.tx_size = v;
        } else {
            self.b.tx_size = max_tx.min(TX_MODE_TO_BIGGEST_TX_SIZE[self.h.tx_mode as usize]);
        }
    }

    fn inter_frame_mode_info(&mut self, d: &mut BoolDecoder) -> Result<()> {
        self.inter_segment_id(d);
        self.read_skip(d);
        self.read_is_inter(d);
        let allow = !self.b.skip || !self.b.is_inter;
        self.read_tx_size(d, allow);
        if self.b.is_inter {
            self.inter_block_mode_info(d)?;
        } else {
            self.intra_block_mode_info(d);
        }
        Ok(())
    }

    fn get_segment_id(&self) -> u8 {
        let bw = NUM_8X8_WIDE[self.b.mi_size as usize] as u32;
        let bh = NUM_8X8_HIGH[self.b.mi_size as usize] as u32;
        let xmis = (self.mi_cols - self.b.mi_col).min(bw);
        let ymis = (self.mi_rows - self.b.mi_row).min(bh);
        let mut seg = 7u8;
        for y in 0..ymis {
            for x in 0..xmis {
                let i = ((self.b.mi_row + y) * self.mi_cols + self.b.mi_col + x) as usize;
                seg = seg.min(self.prev_segment_ids.get(i).copied().unwrap_or(0));
            }
        }
        seg
    }

    fn inter_segment_id(&mut self, d: &mut BoolDecoder) {
        if !self.seg.enabled {
            self.b.segment_id = 0;
            return;
        }
        let predicted = self.get_segment_id();
        if !self.seg.update_map {
            self.b.segment_id = predicted;
            return;
        }
        if self.seg.temporal_update {
            let ctx = (self.left_seg_pred[self.b.mi_row as usize]
                + self.above_seg_pred[self.b.mi_col as usize]) as usize;
            let pred = d.read(self.seg.pred_probs[ctx]);
            self.b.segment_id = if pred {
                predicted
            } else {
                d.tree(&SEGMENT_TREE, |n| self.seg.tree_probs[n])
            };
            let bw = NUM_8X8_WIDE[self.b.mi_size as usize] as usize;
            let bh = NUM_8X8_HIGH[self.b.mi_size as usize] as usize;
            for i in 0..bw {
                self.above_seg_pred[self.b.mi_col as usize + i] = pred as u8;
            }
            for i in 0..bh {
                self.left_seg_pred[self.b.mi_row as usize + i] = pred as u8;
            }
        } else {
            self.b.segment_id = d.tree(&SEGMENT_TREE, |n| self.seg.tree_probs[n]);
        }
    }

    pub(crate) fn left_ref(&self) -> [i8; 2] {
        self.b.left.map_or([INTRA_FRAME, NONE], |m| m.ref_frame)
    }

    pub(crate) fn above_ref(&self) -> [i8; 2] {
        self.b.above.map_or([INTRA_FRAME, NONE], |m| m.ref_frame)
    }

    fn read_is_inter(&mut self, d: &mut BoolDecoder) {
        if self.seg.active(self.b.segment_id, SEG_LVL_REF_FRAME) {
            self.b.is_inter = self.seg.feature_data[self.b.segment_id as usize][SEG_LVL_REF_FRAME]
                != INTRA_FRAME as i16;
        } else {
            let left_intra = self.left_ref()[0] <= INTRA_FRAME;
            let above_intra = self.above_ref()[0] <= INTRA_FRAME;
            let (au, al) = (self.b.avail_u, self.b.avail_l);
            let ctx = if au && al {
                if left_intra && above_intra {
                    3
                } else {
                    (left_intra || above_intra) as usize
                }
            } else if au || al {
                2 * (if au { above_intra } else { left_intra }) as usize
            } else {
                0
            };
            self.b.is_inter = d.read(self.probs.is_inter[ctx]);
            self.counts.is_inter[ctx][self.b.is_inter as usize] += 1;
        }
    }

    fn intra_block_mode_info(&mut self, d: &mut BoolDecoder) {
        self.b.ref_frame = [INTRA_FRAME, NONE];
        if self.b.mi_size >= BLOCK_8X8 {
            let ctx = SIZE_GROUP[self.b.mi_size as usize] as usize;
            let p = self.probs.y_mode[ctx];
            let mode = d.tree(&INTRA_MODE_TREE, |n| p[n]);
            self.counts.intra_mode[ctx][mode as usize] += 1;
            self.b.y_mode = mode;
            self.b.sub_modes = [mode; 4];
        } else {
            let n4w = NUM_4X4_WIDE[self.b.mi_size as usize] as usize;
            let n4h = NUM_4X4_HIGH[self.b.mi_size as usize] as usize;
            let p = self.probs.y_mode[0];
            let mut mode = DC_PRED;
            let mut idy = 0;
            while idy < 2 {
                let mut idx = 0;
                while idx < 2 {
                    mode = d.tree(&INTRA_MODE_TREE, |n| p[n]);
                    self.counts.intra_mode[0][mode as usize] += 1;
                    for y2 in 0..n4h {
                        for x2 in 0..n4w {
                            self.b.sub_modes[(idy + y2) * 2 + idx + x2] = mode;
                        }
                    }
                    idx += n4w;
                }
                idy += n4h;
            }
            self.b.y_mode = mode;
        }
        let p = self.probs.uv_mode[self.b.y_mode as usize];
        let uv = d.tree(&INTRA_MODE_TREE, |n| p[n]);
        self.counts.uv_mode[self.b.y_mode as usize][uv as usize] += 1;
        self.b.uv_mode = uv;
    }

    fn inter_block_mode_info(&mut self, d: &mut BoolDecoder) -> Result<()> {
        self.read_ref_frames(d)?;
        for j in 0..2 {
            if self.b.ref_frame[j] > INTRA_FRAME {
                self.find_best_ref_mvs(j);
            }
        }
        let is_compound = self.b.ref_frame[1] > INTRA_FRAME;
        if self.seg.active(self.b.segment_id, SEG_LVL_SKIP) {
            self.b.y_mode = ZEROMV;
        } else if self.b.mi_size >= BLOCK_8X8 {
            self.b.y_mode = NEARESTMV + self.read_inter_mode(d);
        }
        if self.h.interpolation_filter == SWITCHABLE {
            let left = self.left_ref();
            let above = self.above_ref();
            let left_interp = if self.b.avail_l && left[0] > INTRA_FRAME {
                self.b.left.unwrap().interp_filter
            } else {
                3
            };
            let above_interp = if self.b.avail_u && above[0] > INTRA_FRAME {
                self.b.above.unwrap().interp_filter
            } else {
                3
            };
            let ctx = if left_interp == above_interp {
                left_interp
            } else if left_interp == 3 && above_interp != 3 {
                above_interp
            } else if left_interp != 3 && above_interp == 3 {
                left_interp
            } else {
                3
            } as usize;
            let p = self.probs.interp_filter[ctx];
            let f = d.tree(&INTERP_FILTER_TREE, |n| p[n]);
            self.counts.interp_filter[ctx][f as usize] += 1;
            self.b.interp_filter = f;
        } else {
            self.b.interp_filter = self.h.interpolation_filter;
        }
        if self.b.mi_size < BLOCK_8X8 {
            let n4w = NUM_4X4_WIDE[self.b.mi_size as usize] as usize;
            let n4h = NUM_4X4_HIGH[self.b.mi_size as usize] as usize;
            let mut idy = 0;
            while idy < 2 {
                let mut idx = 0;
                while idx < 2 {
                    self.b.y_mode = NEARESTMV + self.read_inter_mode(d);
                    if self.b.y_mode == NEARESTMV || self.b.y_mode == NEARMV {
                        for j in 0..1 + is_compound as usize {
                            self.append_sub8x8_mvs((idy * 2 + idx) as i32, j);
                        }
                    }
                    self.assign_mv(d, is_compound);
                    for y2 in 0..n4h {
                        for x2 in 0..n4w {
                            let block = (idy + y2) * 2 + idx + x2;
                            for rl in 0..1 + is_compound as usize {
                                self.b.block_mvs[rl][block] = self.b.mv[rl];
                            }
                        }
                    }
                    idx += n4w;
                }
                idy += n4h;
            }
        } else {
            self.assign_mv(d, is_compound);
            for rl in 0..1 + is_compound as usize {
                self.b.block_mvs[rl] = [self.b.mv[rl]; 4];
            }
        }
        Ok(())
    }

    fn read_inter_mode(&mut self, d: &mut BoolDecoder) -> u8 {
        let ctx = self.b.mode_context[self.b.ref_frame[0] as usize] as usize;
        let p = self.probs.inter_mode[ctx];
        let v = d.tree(&INTER_MODE_TREE, |n| p[n]);
        self.counts.inter_mode[ctx][v as usize] += 1;
        v
    }

    fn read_ref_frames(&mut self, d: &mut BoolDecoder) -> Result<()> {
        if self.seg.active(self.b.segment_id, SEG_LVL_REF_FRAME) {
            let rf = self.seg.feature_data[self.b.segment_id as usize][SEG_LVL_REF_FRAME];
            if !(1..=3).contains(&rf) {
                return Err(Error::bitstream(
                    "segment reference feature names no inter frame",
                ));
            }
            self.b.ref_frame = [rf as i8, NONE];
            return Ok(());
        }
        let comp_mode = if self.h.reference_mode == REFERENCE_MODE_SELECT {
            let ctx = self.comp_mode_ctx();
            let v = d.read(self.probs.comp_mode[ctx]);
            self.counts.comp_mode[ctx][v as usize] += 1;
            if v {
                COMPOUND_REFERENCE
            } else {
                SINGLE_REFERENCE
            }
        } else {
            self.h.reference_mode
        };
        if comp_mode == COMPOUND_REFERENCE {
            let idx = self.h.ref_frame_sign_bias[self.h.comp_fixed_ref as usize] as usize;
            let ctx = self.comp_ref_ctx();
            let v = d.read(self.probs.comp_ref[ctx]);
            self.counts.comp_ref[ctx][v as usize] += 1;
            self.b.ref_frame[idx] = self.h.comp_fixed_ref;
            self.b.ref_frame[1 - idx] = self.h.comp_var_ref[v as usize];
        } else {
            let ctx = self.single_ref_p1_ctx();
            let p1 = d.read(self.probs.single_ref[ctx][0]);
            self.counts.single_ref[ctx][0][p1 as usize] += 1;
            if p1 {
                let ctx = self.single_ref_p2_ctx();
                let p2 = d.read(self.probs.single_ref[ctx][1]);
                self.counts.single_ref[ctx][1][p2 as usize] += 1;
                self.b.ref_frame[0] = if p2 { ALTREF_FRAME } else { GOLDEN_FRAME };
            } else {
                self.b.ref_frame[0] = LAST_FRAME;
            }
            self.b.ref_frame[1] = NONE;
        }
        Ok(())
    }

    fn comp_mode_ctx(&self) -> usize {
        let (au, al) = (self.b.avail_u, self.b.avail_l);
        let a = self.above_ref();
        let l = self.left_ref();
        let a_single = a[1] <= NONE;
        let l_single = l[1] <= NONE;
        let a_intra = a[0] <= INTRA_FRAME;
        let l_intra = l[0] <= INTRA_FRAME;
        let fixed = self.h.comp_fixed_ref;
        if au && al {
            if a_single && l_single {
                ((a[0] == fixed) ^ (l[0] == fixed)) as usize
            } else if a_single {
                2 + (a[0] == fixed || a_intra) as usize
            } else if l_single {
                2 + (l[0] == fixed || l_intra) as usize
            } else {
                4
            }
        } else if au {
            if a_single {
                (a[0] == fixed) as usize
            } else {
                3
            }
        } else if al {
            if l_single {
                (l[0] == fixed) as usize
            } else {
                3
            }
        } else {
            1
        }
    }

    fn comp_ref_ctx(&self) -> usize {
        let (au, al) = (self.b.avail_u, self.b.avail_l);
        let a = self.above_ref();
        let l = self.left_ref();
        let a_single = a[1] <= NONE;
        let l_single = l[1] <= NONE;
        let a_intra = a[0] <= INTRA_FRAME;
        let l_intra = l[0] <= INTRA_FRAME;
        let fix_ref_idx = self.h.ref_frame_sign_bias[self.h.comp_fixed_ref as usize] as usize;
        let var_ref_idx = 1 - fix_ref_idx;
        let var1 = self.h.comp_var_ref[1];
        let var0 = self.h.comp_var_ref[0];
        let fixed = self.h.comp_fixed_ref;
        if au && al {
            if a_intra && l_intra {
                2
            } else if l_intra {
                if a_single {
                    1 + 2 * (a[0] != var1) as usize
                } else {
                    1 + 2 * (a[var_ref_idx] != var1) as usize
                }
            } else if a_intra {
                if l_single {
                    1 + 2 * (l[0] != var1) as usize
                } else {
                    1 + 2 * (l[var_ref_idx] != var1) as usize
                }
            } else {
                let vrfa = if a_single { a[0] } else { a[var_ref_idx] };
                let vrfl = if l_single { l[0] } else { l[var_ref_idx] };
                if vrfa == vrfl && var1 == vrfa {
                    0
                } else if l_single && a_single {
                    if (vrfa == fixed && vrfl == var0) || (vrfl == fixed && vrfa == var0) {
                        4
                    } else if vrfa == vrfl {
                        3
                    } else {
                        1
                    }
                } else if l_single || a_single {
                    let vrfc = if l_single { vrfa } else { vrfl };
                    let rfs = if a_single { vrfa } else { vrfl };
                    if vrfc == var1 && rfs != var1 {
                        1
                    } else if rfs == var1 && vrfc != var1 {
                        2
                    } else {
                        4
                    }
                } else if vrfa == vrfl {
                    4
                } else {
                    2
                }
            }
        } else if au {
            if a_intra {
                2
            } else if a_single {
                3 * (a[0] != var1) as usize
            } else {
                4 * (a[var_ref_idx] != var1) as usize
            }
        } else if al {
            if l_intra {
                2
            } else if l_single {
                3 * (l[0] != var1) as usize
            } else {
                4 * (l[var_ref_idx] != var1) as usize
            }
        } else {
            2
        }
    }

    pub(crate) fn single_ref_p1_ctx(&self) -> usize {
        let (au, al) = (self.b.avail_u, self.b.avail_l);
        let a = self.above_ref();
        let l = self.left_ref();
        let a_single = a[1] <= NONE;
        let l_single = l[1] <= NONE;
        let a_intra = a[0] <= INTRA_FRAME;
        let l_intra = l[0] <= INTRA_FRAME;
        const LF: i8 = LAST_FRAME;
        if au && al {
            if a_intra && l_intra {
                2
            } else if l_intra {
                if a_single {
                    4 * (a[0] == LF) as usize
                } else {
                    1 + (a[0] == LF || a[1] == LF) as usize
                }
            } else if a_intra {
                if l_single {
                    4 * (l[0] == LF) as usize
                } else {
                    1 + (l[0] == LF || l[1] == LF) as usize
                }
            } else if a_single && l_single {
                2 * (a[0] == LF) as usize + 2 * (l[0] == LF) as usize
            } else if !a_single && !l_single {
                1 + (a[0] == LF || a[1] == LF || l[0] == LF || l[1] == LF) as usize
            } else {
                let rfs = if a_single { a[0] } else { l[0] };
                let crf1 = if a_single { l[0] } else { a[0] };
                let crf2 = if a_single { l[1] } else { a[1] };
                if rfs == LF {
                    3 + (crf1 == LF || crf2 == LF) as usize
                } else {
                    (crf1 == LF || crf2 == LF) as usize
                }
            }
        } else if au {
            if a_intra {
                2
            } else if a_single {
                4 * (a[0] == LF) as usize
            } else {
                1 + (a[0] == LF || a[1] == LF) as usize
            }
        } else if al {
            if l_intra {
                2
            } else if l_single {
                4 * (l[0] == LF) as usize
            } else {
                1 + (l[0] == LF || l[1] == LF) as usize
            }
        } else {
            2
        }
    }

    pub(crate) fn single_ref_p2_ctx(&self) -> usize {
        let (au, al) = (self.b.avail_u, self.b.avail_l);
        let a = self.above_ref();
        let l = self.left_ref();
        let a_single = a[1] <= NONE;
        let l_single = l[1] <= NONE;
        let a_intra = a[0] <= INTRA_FRAME;
        let l_intra = l[0] <= INTRA_FRAME;
        const LF: i8 = LAST_FRAME;
        const GF: i8 = GOLDEN_FRAME;
        const AF: i8 = ALTREF_FRAME;
        if au && al {
            if a_intra && l_intra {
                2
            } else if l_intra {
                if a_single {
                    if a[0] == LF {
                        3
                    } else {
                        4 * (a[0] == GF) as usize
                    }
                } else {
                    1 + 2 * (a[0] == GF || a[1] == GF) as usize
                }
            } else if a_intra {
                if l_single {
                    if l[0] == LF {
                        3
                    } else {
                        4 * (l[0] == GF) as usize
                    }
                } else {
                    1 + 2 * (l[0] == GF || l[1] == GF) as usize
                }
            } else if a_single && l_single {
                if a[0] == LF && l[0] == LF {
                    3
                } else if a[0] == LF {
                    4 * (l[0] == GF) as usize
                } else if l[0] == LF {
                    4 * (a[0] == GF) as usize
                } else {
                    2 * (a[0] == GF) as usize + 2 * (l[0] == GF) as usize
                }
            } else if !a_single && !l_single {
                if a[0] == l[0] && a[1] == l[1] {
                    3 * (a[0] == GF || a[1] == GF) as usize
                } else {
                    2
                }
            } else {
                let rfs = if a_single { a[0] } else { l[0] };
                let crf1 = if a_single { l[0] } else { a[0] };
                let crf2 = if a_single { l[1] } else { a[1] };
                if rfs == GF {
                    3 + (crf1 == GF || crf2 == GF) as usize
                } else if rfs == AF {
                    (crf1 == GF || crf2 == GF) as usize
                } else {
                    1 + 2 * (crf1 == GF || crf2 == GF) as usize
                }
            }
        } else if au {
            if a_intra || (a[0] == LF && a_single) {
                2
            } else if a_single {
                4 * (a[0] == GF) as usize
            } else {
                3 * (a[0] == GF || a[1] == GF) as usize
            }
        } else if al {
            if l_intra || (l[0] == LF && l_single) {
                2
            } else if l_single {
                4 * (l[0] == GF) as usize
            } else {
                3 * (l[0] == GF || l[1] == GF) as usize
            }
        } else {
            2
        }
    }

    fn assign_mv(&mut self, d: &mut BoolDecoder, is_compound: bool) {
        self.b.mv[1] = [0, 0];
        for i in 0..1 + is_compound as usize {
            self.b.mv[i] = match self.b.y_mode {
                NEWMV => self.read_mv(d, i),
                NEARESTMV => self.b.nearest_mv[i],
                NEARMV => self.b.near_mv[i],
                _ => [0, 0],
            };
        }
    }

    fn read_mv(&mut self, d: &mut BoolDecoder, r: usize) -> Mv {
        let best = self.b.best_mv[r];
        let use_hp = self.h.allow_high_precision_mv && use_mv_hp(best);
        let mut diff = [0i32; 2];
        let p = self.probs.mv_joint;
        let joint = d.tree(&MV_JOINT_TREE, |n| p[n]);
        self.counts.mv_joint[joint as usize] += 1;
        if joint == MV_JOINT_HZVNZ || joint == MV_JOINT_HNZVNZ {
            diff[0] = self.read_mv_component(d, 0, use_hp);
        }
        if joint == MV_JOINT_HNZVZ || joint == MV_JOINT_HNZVNZ {
            diff[1] = self.read_mv_component(d, 1, use_hp);
        }
        [best[0] + diff[0], best[1] + diff[1]]
    }

    fn read_mv_component(&mut self, d: &mut BoolDecoder, comp: usize, use_hp: bool) -> i32 {
        let sign = d.read(self.probs.mv_sign[comp]);
        self.counts.mv_sign[comp][sign as usize] += 1;
        let pc = self.probs.mv_class[comp];
        let class = d.tree(&MV_CLASS_TREE, |n| pc[n]) as u32;
        self.counts.mv_class[comp][class as usize] += 1;
        let mag = if class == 0 {
            let c0 = d.read(self.probs.mv_class0_bit[comp]) as i32;
            self.counts.mv_class0_bit[comp][c0 as usize] += 1;
            let pf = self.probs.mv_class0_fr[comp][c0 as usize];
            let fr = d.tree(&MV_FR_TREE, |n| pf[n]) as i32;
            self.counts.mv_class0_fr[comp][c0 as usize][fr as usize] += 1;
            let hp = if use_hp {
                d.read(self.probs.mv_class0_hp[comp]) as i32
            } else {
                1
            };
            self.counts.mv_class0_hp[comp][hp as usize] += 1;
            ((c0 << 3) | (fr << 1) | hp) + 1
        } else {
            let mut dd = 0i32;
            for i in 0..class as usize {
                let bit = d.read(self.probs.mv_bits[comp][i]);
                self.counts.mv_bits[comp][i][bit as usize] += 1;
                dd |= (bit as i32) << i;
            }
            let mag = 2 << (class + 2);
            let pf = self.probs.mv_fr[comp];
            let fr = d.tree(&MV_FR_TREE, |n| pf[n]) as i32;
            self.counts.mv_fr[comp][fr as usize] += 1;
            let hp = if use_hp {
                d.read(self.probs.mv_hp[comp]) as i32
            } else {
                1
            };
            self.counts.mv_hp[comp][hp as usize] += 1;
            mag + ((dd << 3) | (fr << 1) | hp) + 1
        };
        if sign { -mag } else { mag }
    }

    // -----------------------------------------------------------------
    // Residual (6.4.21 - 6.4.26).

    fn residual(&mut self, d: &mut BoolDecoder) -> Result<()> {
        let bsize = if self.b.mi_size < BLOCK_8X8 {
            BLOCK_8X8
        } else {
            self.b.mi_size
        };
        for plane in 0..3usize {
            let tx_sz = if plane > 0 {
                self.uv_tx_size()
            } else {
                self.b.tx_size
            };
            let step = 1usize << tx_sz;
            let (sx, sy) = if plane > 0 {
                (self.ss_x, self.ss_y)
            } else {
                (0, 0)
            };
            let plane_sz = SS_SIZE_LOOKUP[bsize as usize][sx as usize][sy as usize];
            if plane_sz == BLOCK_INVALID {
                return Err(Error::bitstream("invalid chroma block size"));
            }
            let n4w = NUM_4X4_WIDE[plane_sz as usize] as usize;
            let n4h = NUM_4X4_HIGH[plane_sz as usize] as usize;
            let base_x = ((self.b.mi_col * 8) >> sx) as usize;
            let base_y = ((self.b.mi_row * 8) >> sy) as usize;
            if self.b.is_inter {
                if self.b.mi_size < BLOCK_8X8 {
                    for y in 0..n4h {
                        for x in 0..n4w {
                            self.predict_inter(
                                plane,
                                base_x + 4 * x,
                                base_y + 4 * y,
                                4,
                                4,
                                y * n4w + x,
                            )?;
                        }
                    }
                } else {
                    self.predict_inter(plane, base_x, base_y, n4w * 4, n4h * 4, 0)?;
                }
            }
            let max_x = ((self.mi_cols * 8) >> sx) as usize;
            let max_y = ((self.mi_rows * 8) >> sy) as usize;
            let mut block_idx = 0;
            let mut y = 0;
            while y < n4h {
                let mut x = 0;
                while x < n4w {
                    let start_x = base_x + 4 * x;
                    let start_y = base_y + 4 * y;
                    let mut nonzero = false;
                    if start_x < max_x && start_y < max_y {
                        if !self.b.is_inter {
                            self.predict_intra(
                                plane,
                                start_x,
                                start_y,
                                self.b.avail_l || x > 0,
                                self.b.avail_u || y > 0,
                                x + step < n4w,
                                tx_sz,
                                block_idx,
                            );
                        }
                        if !self.b.skip {
                            let (eob, tx_type) =
                                self.tokens(d, plane, start_x, start_y, tx_sz, block_idx);
                            nonzero = eob > 0;
                            if eob > 0 {
                                self.reconstruct(plane, start_x, start_y, tx_sz, tx_type, eob);
                            }
                        }
                    }
                    let an = &mut self.above_nonzero[plane];
                    let ln = &mut self.left_nonzero[plane];
                    for i in 0..step {
                        an[(start_x >> 2) + i] = nonzero as u8;
                        ln[(start_y >> 2) + i] = nonzero as u8;
                    }
                    block_idx += 1;
                    x += step;
                }
                y += step;
            }
        }
        Ok(())
    }

    /// get_uv_tx_size() (6.4.22).
    pub(crate) fn uv_tx_size(&self) -> u8 {
        if self.b.mi_size < BLOCK_8X8 {
            return TX_4X4;
        }
        let (sx, sy) = self.uv_tx_subsampling();
        let uv = SS_SIZE_LOOKUP[self.b.mi_size as usize][sx][sy];
        self.b.tx_size.min(MAX_TXSIZE_LOOKUP[uv as usize])
    }

    /// The subsampling get_uv_tx_size and the chroma motion vectors of
    /// blocks below 8x8 use: the frame's, or 4:2:0's for a pre-final
    /// profile 1 stream (`FrameHeader::legacy_uv`).
    pub(crate) fn uv_tx_subsampling(&self) -> (usize, usize) {
        if self.h.legacy_uv {
            (1, 1)
        } else {
            (self.ss_x as usize, self.ss_y as usize)
        }
    }

    /// The TxType of get_scan (6.4.25).
    pub(crate) fn tx_type(&self, plane: usize, tx_sz: u8, block_idx: usize) -> u8 {
        if plane > 0 || tx_sz == TX_32X32 {
            DCT_DCT
        } else if tx_sz == TX_4X4 {
            if self.h.lossless || self.b.is_inter {
                DCT_DCT
            } else {
                MODE2TXFM_MAP[if self.b.mi_size < BLOCK_8X8 {
                    self.b.sub_modes[block_idx]
                } else {
                    self.b.y_mode
                } as usize]
            }
        } else {
            MODE2TXFM_MAP[self.b.y_mode as usize]
        }
    }

    /// tokens() (6.4.24): reads the coefficients of one transform block into
    /// `self.coefs` (in raster order). Returns the end of block and TxType.
    fn tokens(
        &mut self,
        d: &mut BoolDecoder,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: u8,
        block_idx: usize,
    ) -> (usize, u8) {
        let seg_eob = 16usize << (tx_sz << 1);
        let tx_type = self.tx_type(plane, tx_sz, block_idx);
        let scan: &[u16] = match (tx_sz, tx_type) {
            (TX_4X4, ADST_DCT) => &ROW_SCAN_4X4,
            (TX_4X4, DCT_ADST) => &COL_SCAN_4X4,
            (TX_4X4, _) => &DEFAULT_SCAN_4X4,
            (TX_8X8, ADST_DCT) => &ROW_SCAN_8X8,
            (TX_8X8, DCT_ADST) => &COL_SCAN_8X8,
            (TX_8X8, _) => &DEFAULT_SCAN_8X8,
            (TX_16X16, ADST_DCT) => &ROW_SCAN_16X16,
            (TX_16X16, DCT_ADST) => &COL_SCAN_16X16,
            (TX_16X16, _) => &DEFAULT_SCAN_16X16,
            _ => &DEFAULT_SCAN_32X32,
        };
        let ref_type = self.b.is_inter as usize;
        let ptype = (plane > 0) as usize;
        let txs = tx_sz as usize;
        // Context of the first coefficient.
        let (sx, sy) = if plane > 0 {
            (self.ss_x, self.ss_y)
        } else {
            (0, 0)
        };
        let max_x4 = ((2 * self.mi_cols) >> sx) as usize;
        let max_y4 = ((2 * self.mi_rows) >> sy) as usize;
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let mut above = 0u8;
        let mut left = 0u8;
        for i in 0..(1usize << tx_sz) {
            if x4 + i < max_x4 {
                above |= self.above_nonzero[plane][x4 + i];
            }
            if y4 + i < max_y4 {
                left |= self.left_nonzero[plane][y4 + i];
            }
        }
        let mut ctx = (above + left) as usize;
        let n = 4usize << tx_sz;
        let log2n = 2 + tx_sz as u32;
        let mut check_eob = true;
        let mut c = 0usize;
        let bd = self.bit_depth;
        while c < seg_eob {
            let pos = scan[c] as usize;
            let band = if tx_sz == TX_4X4 {
                COEFBAND_4X4[c]
            } else {
                COEFBAND_8X8PLUS[c]
            } as usize;
            if c > 0 {
                // Neighbour context.
                let i = pos >> log2n;
                let j = pos & (n - 1);
                let (a, b) = if i > 0 && j > 0 {
                    let a = (i - 1) * n + j;
                    let a2 = i * n + j - 1;
                    match tx_type {
                        DCT_ADST => (a, a),
                        ADST_DCT => (a2, a2),
                        _ => (a, a2),
                    }
                } else if i > 0 {
                    ((i - 1) * n + j, (i - 1) * n + j)
                } else {
                    (j - 1, j - 1)
                };
                ctx = (1 + self.token_cache[a] as usize + self.token_cache[b] as usize) >> 1;
            }
            let probs = &self.probs.coef[txs][ptype][ref_type][band][ctx];
            if check_eob {
                let more = d.read(probs[0]);
                self.counts.more_coefs[txs][ptype][ref_type][band][ctx][more as usize] += 1;
                if !more {
                    break;
                }
            }
            let token = d.tree(&TOKEN_TREE, |node| pareto(node, probs[(1 + node).min(2)]));
            self.counts.token[txs][ptype][ref_type][band][ctx][(token as usize).min(2)] += 1;
            self.token_cache[pos] = ENERGY_CLASS[token as usize];
            if token == ZERO_TOKEN {
                self.coefs[pos] = 0;
                check_eob = false;
            } else {
                let coef = read_coef(d, token, bd);
                let sign = d.read(128);
                self.coefs[pos] = if sign { -coef } else { coef };
                check_eob = true;
            }
            c += 1;
        }
        if c > 0 {
            self.eob_total += 1;
        }
        (c, tx_type)
    }
}

fn read_coef(d: &mut BoolDecoder, token: u8, bit_depth: u32) -> i32 {
    let [cat, num_extra, base] = EXTRA_BITS[token as usize];
    let mut coef = base;
    if token == DCT_VAL_CAT6 {
        for e in 0..bit_depth as i32 - 8 {
            let high = d.read(255) as i32;
            coef += high << (5 + bit_depth as i32 - e);
        }
    }
    let probs = CAT_PROBS[cat as usize];
    for e in 0..num_extra {
        let bit = d.read(probs[e as usize]) as i32;
        coef += bit << (num_extra - 1 - e);
    }
    coef
}

/// get_tile_offset (6.4.1).
fn tile_offset(tile_num: u32, mis: u32, log2: u32) -> u32 {
    let sbs = (mis + 7) >> 3;
    let offset = ((tile_num * sbs) >> log2) << 3;
    offset.min(mis)
}

/// use_mv_hp (6.5.13).
pub(crate) fn use_mv_hp(mv: Mv) -> bool {
    (mv[0].abs() >> 3) < COMPANDED_MVREF_THRESH && (mv[1].abs() >> 3) < COMPANDED_MVREF_THRESH
}
