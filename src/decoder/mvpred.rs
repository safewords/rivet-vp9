//! Motion vector prediction (6.5).

use super::block::{FrameDec, Mv, use_mv_hp};
use crate::consts::*;
use crate::tables::*;

/// The working state of find_mv_refs.
struct Search {
    count: usize,
    list: [Mv; 2],
    cand_mv: [Mv; 2],
    cand_frame: [i8; 2],
}

impl Search {
    /// add_mv_ref_list( refList ) (6.5.6).
    fn add(&mut self, ref_list: usize) {
        if self.count >= 2 {
            return;
        }
        if self.count > 0 && self.cand_mv[ref_list] == self.list[0] {
            return;
        }
        self.list[self.count] = self.cand_mv[ref_list];
        self.count += 1;
    }
}

impl FrameDec<'_> {
    /// is_inside (6.5.2).
    #[inline]
    fn is_inside(&self, r: i32, c: i32) -> bool {
        r >= 0 && r < self.mi_rows as i32 && c >= self.mi_col_start as i32 && c < self.mi_col_end as i32
    }

    /// get_block_mv (6.5.10).
    fn get_block_mv(&self, s: &mut Search, r: i32, c: i32, ref_list: usize, use_prev: bool) {
        let idx = (r as u32 * self.mi_cols + c as u32) as usize;
        if use_prev {
            let p = self.prev_mvs.unwrap()[idx];
            s.cand_mv[ref_list] = p.mv[ref_list];
            s.cand_frame[ref_list] = p.ref_frame[ref_list];
        } else {
            let m = &self.mi[idx];
            s.cand_mv[ref_list] = m.mv[ref_list][3];
            s.cand_frame[ref_list] = m.ref_frame[ref_list];
        }
    }

    /// if_same_ref_frame_add_mv (6.5.7).
    fn if_same_ref_frame_add(&self, s: &mut Search, r: i32, c: i32, ref_frame: i8, use_prev: bool) {
        for j in 0..2 {
            self.get_block_mv(s, r, c, j, use_prev);
            if s.cand_frame[j] == ref_frame {
                s.add(j);
                return;
            }
        }
    }

    /// scale_mv (6.5.9).
    fn scale_mv(&self, s: &mut Search, ref_list: usize, ref_frame: i8) {
        let cand = s.cand_frame[ref_list];
        if self.h.ref_frame_sign_bias[cand as usize] != self.h.ref_frame_sign_bias[ref_frame as usize] {
            s.cand_mv[ref_list][0] *= -1;
            s.cand_mv[ref_list][1] *= -1;
        }
    }

    /// if_diff_ref_frame_add_mv (6.5.8).
    fn if_diff_ref_frame_add(&self, s: &mut Search, r: i32, c: i32, ref_frame: i8, use_prev: bool) {
        for j in 0..2 {
            self.get_block_mv(s, r, c, j, use_prev);
        }
        let mvs_same = s.cand_mv[0] == s.cand_mv[1];
        if s.cand_frame[0] > INTRA_FRAME && s.cand_frame[0] != ref_frame {
            self.scale_mv(s, 0, ref_frame);
            s.add(0);
        }
        if s.cand_frame[1] > INTRA_FRAME && s.cand_frame[1] != ref_frame && !mvs_same {
            self.scale_mv(s, 1, ref_frame);
            s.add(1);
        }
    }

    fn clamp_mv_row(&self, mvec: i32, border: i32) -> i32 {
        let bh = NUM_8X8_HIGH[self.b.mi_size as usize] as i32;
        let mi_row = self.b.mi_row as i32;
        let to_top = -((mi_row * 8) * 8);
        let to_bottom = ((self.mi_rows as i32 - bh - mi_row) * 8) * 8;
        mvec.clamp(to_top - border, to_bottom + border)
    }

    fn clamp_mv_col(&self, mvec: i32, border: i32) -> i32 {
        let bw = NUM_8X8_WIDE[self.b.mi_size as usize] as i32;
        let mi_col = self.b.mi_col as i32;
        let to_left = -((mi_col * 8) * 8);
        let to_right = ((self.mi_cols as i32 - bw - mi_col) * 8) * 8;
        mvec.clamp(to_left - border, to_right + border)
    }

    /// find_mv_refs( refFrame, block ) (6.5.1). Returns RefListMv.
    pub(crate) fn find_mv_refs(&mut self, ref_frame: i8, block: i32) -> [Mv; 2] {
        let mut s = Search { count: 0, list: [[0; 2]; 2], cand_mv: [[0; 2]; 2], cand_frame: [0; 2] };
        let mut context_counter = 0usize;
        let search = &MV_REF_BLOCKS[self.b.mi_size as usize];
        let (mi_row, mi_col) = (self.b.mi_row as i32, self.b.mi_col as i32);
        for cand in search.iter().take(2) {
            let r = mi_row + cand[0] as i32;
            let c = mi_col + cand[1] as i32;
            if self.is_inside(r, c) {
                let m = self.mi[(r as u32 * self.mi_cols + c as u32) as usize];
                context_counter += MODE_2_COUNTER[m.y_mode as usize] as usize;
                for j in 0..2 {
                    if m.ref_frame[j] == ref_frame {
                        // get_sub_block_mv (6.5.11)
                        let idx = if block >= 0 { IDX_N_COLUMN_TO_SUBBLOCK[block as usize][(cand[1] == 0) as usize] as usize } else { 3 };
                        s.cand_mv[j] = m.mv[j][idx];
                        s.add(j);
                        break;
                    }
                }
            }
        }
        for cand in search.iter().skip(2) {
            let r = mi_row + cand[0] as i32;
            let c = mi_col + cand[1] as i32;
            if self.is_inside(r, c) {
                self.if_same_ref_frame_add(&mut s, r, c, ref_frame, false);
            }
        }
        let use_prev = self.prev_mvs.is_some();
        if use_prev {
            self.if_same_ref_frame_add(&mut s, mi_row, mi_col, ref_frame, true);
        }
        for cand in search.iter() {
            let r = mi_row + cand[0] as i32;
            let c = mi_col + cand[1] as i32;
            if self.is_inside(r, c) {
                self.if_diff_ref_frame_add(&mut s, r, c, ref_frame, false);
            }
        }
        if use_prev {
            self.if_diff_ref_frame_add(&mut s, mi_row, mi_col, ref_frame, true);
        }
        self.b.mode_context[ref_frame as usize] = COUNTER_TO_CONTEXT[context_counter];
        for i in 0..2 {
            s.list[i][0] = self.clamp_mv_row(s.list[i][0], MV_BORDER);
            s.list[i][1] = self.clamp_mv_col(s.list[i][1], MV_BORDER);
        }
        s.list
    }

    /// find_mv_refs then find_best_ref_mvs( refList ) (6.5.12), for the
    /// block's reference `ref_list`.
    pub(crate) fn find_best_ref_mvs(&mut self, ref_list: usize) {
        let mut list = self.find_mv_refs_cached(ref_list);
        let border = (BORDERINPIXELS - INTERP_EXTEND) << 3;
        for mv in list.iter_mut() {
            let (mut dr, mut dc) = (mv[0], mv[1]);
            if !self.h.allow_high_precision_mv || !use_mv_hp(*mv) {
                if dr & 1 != 0 {
                    dr += if dr > 0 { -1 } else { 1 };
                }
                if dc & 1 != 0 {
                    dc += if dc > 0 { -1 } else { 1 };
                }
            }
            mv[0] = self.clamp_mv_row(dr, border);
            mv[1] = self.clamp_mv_col(dc, border);
        }
        self.b.nearest_mv[ref_list] = list[0];
        self.b.near_mv[ref_list] = list[1];
        self.b.best_mv[ref_list] = list[0];
    }

    fn find_mv_refs_cached(&mut self, ref_list: usize) -> [Mv; 2] {
        self.find_mv_refs(self.b.ref_frame[ref_list], -1)
    }

    /// append_sub8x8_mvs( block, refList ) (6.5.14).
    pub(crate) fn append_sub8x8_mvs(&mut self, block: i32, ref_list: usize) {
        let list = self.find_mv_refs(self.b.ref_frame[ref_list], block);
        let mut sub: [Mv; 2] = [[0; 2]; 2];
        let mut dst = 0usize;
        let bm = self.b.block_mvs[ref_list];
        if block == 0 {
            sub = list;
            dst = 2;
        } else if block <= 2 {
            sub[dst] = bm[0];
            dst += 1;
        } else {
            sub[dst] = bm[2];
            dst += 1;
            let mut idx = 1i32;
            while idx >= 0 && dst < 2 {
                if bm[idx as usize] != sub[0] {
                    sub[dst] = bm[idx as usize];
                    dst += 1;
                }
                idx -= 1;
            }
        }
        let mut n = 0;
        while n < 2 && dst < 2 {
            if list[n] != sub[0] {
                sub[dst] = list[n];
                dst += 1;
            }
            n += 1;
        }
        if dst < 2 {
            sub[dst] = [0, 0];
        }
        self.b.nearest_mv[ref_list] = sub[0];
        self.b.near_mv[ref_list] = sub[1];
    }
}
