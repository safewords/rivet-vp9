//! Block decisions and the tile syntax writer.
//!
//! The decoder's `FrameDec` is the reconstruction engine: its prediction,
//! motion vector prediction and reconstruct functions are called exactly as
//! the residual syntax calls them, on its picture buffers, so the encoder's
//! picture is the decoder's picture.

// Loops index arrays the way the specification's formulas do.
#![allow(clippy::needless_range_loop)]

use crate::bool_coder::BoolEncoder;
use crate::consts::*;
use crate::decoder::block::{Block, Mv, pareto};
use crate::decoder::{FrameDec, MiInfo, RefFrame};
use crate::dsp::inter;
use crate::frame::Frame;
use crate::header::FrameHeader;
use crate::tables::*;

use super::Config;
use super::fdct;

/// The source picture, padded to whole superblocks by repeating its edges.
pub(crate) struct Source {
    pub planes: [Vec<u16>; 3],
    pub stride: [usize; 3],
}

impl Source {
    pub(crate) fn new(f: &Frame, h: &FrameHeader) -> Self {
        let w = (h.sb64_cols * 64) as usize;
        let ht = (h.sb64_rows * 64) as usize;
        let mut planes: [Vec<u16>; 3] = Default::default();
        let mut stride = [0; 3];
        for p in 0..3 {
            let (pw, ph) = if p == 0 { (w, ht) } else { (w >> 1, ht >> 1) };
            let fp = f.planes[p];
            let mut v = vec![0u16; pw * ph];
            for y in 0..ph {
                let sy = y.min(fp.height as usize - 1);
                for x in 0..pw {
                    let sx = x.min(fp.width as usize - 1);
                    v[y * pw + x] = f.sample(p, sx as u32, sy as u32);
                }
            }
            planes[p] = v;
            stride[p] = pw;
        }
        Source { planes, stride }
    }
}

/// One coded transform block.
struct TxBlock {
    coefs: Vec<i32>,
    tx_type: u8,
    eob: usize,
}

/// The transform blocks of one plane of a block, in syntax order; `None`
/// for those outside the frame.
type PlaneCoding = Vec<Option<TxBlock>>;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Choice {
    Intra { y_mode: u8, uv_mode: u8 },
    Inter { y_mode: u8, mv: Mv },
}

pub(crate) struct TileEncoder<'a> {
    cfg: &'a Config,
    h: &'a FrameHeader,
    src: &'a Source,
    /// The LAST frame, its luma padded for whole-pixel search.
    last: Option<&'a RefFrame>,
    padded: Vec<u16>,
    pad_stride: usize,
    /// Rate-distortion multiplier (squared error per bit).
    lambda: f64,
    /// How often each of the first three coefficient probabilities of
    /// every context saw a 0 and a 1: what forward updates are judged by.
    pub(crate) stats: Box<CoefStats>,
}

/// `[txSz][plane > 0][is_inter][band][ctx][node][bit]`.
pub(crate) type CoefStats = [[[[[[[u32; 2]; 3]; 6]; 6]; 2]; 2]; 4];

const PAD: usize = 96;

fn tree_bits(tree: &[i8], value: u8, prob: impl Fn(usize) -> u8) -> f64 {
    fn path(tree: &[i8], node: usize, value: u8, out: &mut Vec<(usize, bool)>) -> bool {
        for b in 0..2 {
            let t = tree[node + b];
            out.push((node, b == 1));
            if t <= 0 {
                if (-t) as u8 == value {
                    return true;
                }
            } else if path(tree, t as usize, value, out) {
                return true;
            }
            out.pop();
        }
        false
    }
    let mut p = Vec::new();
    path(tree, 0, value, &mut p);
    p.iter().map(|&(n, b)| bool_bits(prob(n >> 1), b)).sum()
}

fn bool_bits(p: u8, bit: bool) -> f64 {
    let p0 = p as f64 / 256.0;
    -(if bit { 1.0 - p0 } else { p0 }).log2()
}

fn mv_bits(d: Mv) -> f64 {
    let comp = |v: i32| {
        if v == 0 {
            0.0
        } else {
            3.0 + 2.0 * ((v.unsigned_abs() as f64) / 8.0 + 1.0).log2()
        }
    };
    2.0 + comp(d[0]) + comp(d[1])
}

impl<'a> TileEncoder<'a> {
    pub(crate) fn new(
        cfg: &'a Config,
        h: &'a FrameHeader,
        src: &'a Source,
        last: Option<&'a RefFrame>,
    ) -> Self {
        let q = AC_QLOOKUP[0][h.base_q_idx as usize] as f64 / 8.0;
        let lambda = 0.12 * q * q;
        let mut te = TileEncoder {
            cfg,
            h,
            src,
            last,
            padded: Vec::new(),
            pad_stride: 0,
            lambda,
            stats: Box::new([[[[[[[0; 2]; 3]; 6]; 6]; 2]; 2]; 4]),
        };
        if let Some(r) = last {
            let w = r.width as usize;
            let ht = r.height as usize;
            let ps = (h.sb64_cols * 64) as usize + 2 * PAD;
            let rows = (h.sb64_rows * 64) as usize + 2 * PAD;
            let mut v = vec![0u16; ps * rows];
            let rp = &r.planes[0];
            for y in 0..rows {
                let sy = (y as isize - PAD as isize).clamp(0, ht as isize - 1) as usize;
                for x in 0..ps {
                    let sx = (x as isize - PAD as isize).clamp(0, w as isize - 1) as usize;
                    v[y * ps + x] = rp.data[sy * rp.stride + sx];
                }
            }
            te.padded = v;
            te.pad_stride = ps;
        }
        te
    }

    pub(crate) fn encode_tiles(&mut self, fd: &mut FrameDec) -> Vec<u8> {
        let h = self.h;
        let tile_cols = 1u32 << h.tile_cols_log2;
        for p in fd.above_nonzero.iter_mut() {
            p.iter_mut().for_each(|v| *v = 0);
        }
        fd.above_partition.iter_mut().for_each(|v| *v = 0);
        fd.above_seg_pred.iter_mut().for_each(|v| *v = 0);
        let mut out = Vec::new();
        for tile_col in 0..tile_cols {
            let off = |n: u32| {
                let sbs = h.mi_cols.div_ceil(8);
                (((n * sbs) >> h.tile_cols_log2) << 3).min(h.mi_cols)
            };
            fd.mi_row_start = 0;
            fd.mi_row_end = h.mi_rows;
            fd.mi_col_start = off(tile_col);
            fd.mi_col_end = off(tile_col + 1);
            let mut e = BoolEncoder::new();
            let mut r = 0;
            while r < h.mi_rows {
                for p in fd.left_nonzero.iter_mut() {
                    p.iter_mut().for_each(|v| *v = 0);
                }
                fd.left_partition.iter_mut().for_each(|v| *v = 0);
                fd.left_seg_pred.iter_mut().for_each(|v| *v = 0);
                let mut c = fd.mi_col_start;
                while c < fd.mi_col_end {
                    self.partition(&mut e, fd, r, c, BLOCK_64X64);
                    c += 8;
                }
                r += 8;
            }
            let data = e.finish();
            if tile_col + 1 < tile_cols {
                out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            }
            out.extend_from_slice(&data);
        }
        out
    }

    fn target_size(&self) -> u8 {
        match self.cfg.block_size {
            8 => BLOCK_8X8,
            16 => BLOCK_16X16,
            32 => BLOCK_32X32,
            _ => BLOCK_64X64,
        }
    }

    fn partition(&mut self, e: &mut BoolEncoder, fd: &mut FrameDec, r: u32, c: u32, bsize: u8) {
        let h = self.h;
        if r >= h.mi_rows || c >= h.mi_cols {
            return;
        }
        let num8x8 = NUM_8X8_WIDE[bsize as usize] as u32;
        let half = num8x8 >> 1;
        let has_rows = (r + half) < h.mi_rows;
        let has_cols = (c + half) < h.mi_cols;
        let partition = if bsize > self.target_size() {
            PARTITION_SPLIT
        } else if has_rows && has_cols {
            PARTITION_NONE
        } else if has_cols {
            PARTITION_HORZ
        } else if has_rows {
            PARTITION_VERT
        } else {
            PARTITION_SPLIT
        };
        // The symbol, with the decoder's context.
        let bsl = MI_WIDTH_LOG2[bsize as usize] as u32;
        let boffset = 3 - bsl;
        let mut above = 0u8;
        let mut left = 0u8;
        for i in 0..num8x8 {
            above |= fd.above_partition[(c + i) as usize];
            left |= fd.left_partition[(r + i) as usize];
        }
        let ctx = (bsl * 4 + (((left >> boffset) & 1) as u32) * 2 + ((above >> boffset) & 1) as u32)
            as usize;
        let probs = if h.frame_is_intra {
            KF_PARTITION_PROBS[ctx]
        } else {
            fd.probs.partition[ctx]
        };
        if has_rows && has_cols {
            e.tree(&PARTITION_TREE, partition, |n| probs[n]);
        } else if has_cols {
            e.write(partition == PARTITION_SPLIT, probs[1]);
        } else if has_rows {
            e.write(partition == PARTITION_SPLIT, probs[2]);
        }
        let subsize = SUBSIZE_LOOKUP[partition as usize][bsize as usize];
        if subsize < BLOCK_8X8 || partition == PARTITION_NONE {
            self.block(e, fd, r, c, subsize);
        } else if partition == PARTITION_HORZ {
            self.block(e, fd, r, c, subsize);
            if has_rows {
                self.block(e, fd, r + half, c, subsize);
            }
        } else if partition == PARTITION_VERT {
            self.block(e, fd, r, c, subsize);
            if has_cols {
                self.block(e, fd, r, c + half, subsize);
            }
        } else {
            self.partition(e, fd, r, c, subsize);
            self.partition(e, fd, r, c + half, subsize);
            self.partition(e, fd, r + half, c, subsize);
            self.partition(e, fd, r + half, c + half, subsize);
        }
        if bsize == BLOCK_8X8 || partition != PARTITION_SPLIT {
            let a = 15 >> B_WIDTH_LOG2[subsize as usize];
            let l = 15 >> B_HEIGHT_LOG2[subsize as usize];
            for i in 0..num8x8 {
                fd.above_partition[(c + i) as usize] = a;
                fd.left_partition[(r + i) as usize] = l;
            }
        }
    }

    // -----------------------------------------------------------------
    // Coding a block's planes.

    /// The plane region of the current block: (x, y, w, h, plane block size).
    fn plane_region(&self, fd: &FrameDec, plane: usize) -> (usize, usize, usize, usize, u8) {
        let (sx, sy) = if plane > 0 { (1, 1) } else { (0, 0) };
        let bsize = fd.b.mi_size.max(BLOCK_8X8);
        let psz = SS_SIZE_LOOKUP[bsize as usize][sx][sy];
        let x = ((fd.b.mi_col * 8) >> sx) as usize;
        let y = ((fd.b.mi_row * 8) >> sy) as usize;
        (
            x,
            y,
            NUM_4X4_WIDE[psz as usize] as usize * 4,
            NUM_4X4_HIGH[psz as usize] as usize * 4,
            psz,
        )
    }

    fn save(&self, fd: &FrameDec, plane: usize) -> Vec<u16> {
        let (x, y, w, h, _) = self.plane_region(fd, plane);
        let b = &fd.planes[plane];
        let mut v = Vec::with_capacity(w * h);
        for i in 0..h {
            v.extend_from_slice(&b.data[(y + i) * b.stride + x..(y + i) * b.stride + x + w]);
        }
        v
    }

    fn restore(&self, fd: &mut FrameDec, plane: usize, saved: &[u16]) {
        let (x, y, w, h, _) = self.plane_region(fd, plane);
        let b = &mut fd.planes[plane];
        for i in 0..h {
            b.data[(y + i) * b.stride + x..(y + i) * b.stride + x + w]
                .copy_from_slice(&saved[i * w..(i + 1) * w]);
        }
    }

    /// Squared error of the reconstruction against the source over the
    /// visible part of the plane region.
    fn sse(&self, fd: &FrameDec, plane: usize) -> f64 {
        let (x, y, w, h, _) = self.plane_region(fd, plane);
        let (sx, sy) = if plane > 0 { (1, 1) } else { (0, 0) };
        let max_x = ((self.h.mi_cols as usize * 8) >> sx).min(x + w);
        let max_y = ((self.h.mi_rows as usize * 8) >> sy).min(y + h);
        let b = &fd.planes[plane];
        let s = &self.src.planes[plane];
        let ss = self.src.stride[plane];
        let mut acc = 0u64;
        for yy in y..max_y {
            for xx in x..max_x {
                let d = b.data[yy * b.stride + xx] as i64 - s[yy * ss + xx] as i64;
                acc += (d * d) as u64;
            }
        }
        acc as f64
    }

    /// Runs the residual syntax's loop over one plane of the current block:
    /// predicts (intra, per transform block), transforms, quantises and
    /// reconstructs. Inter prediction must already be in the buffer.
    /// Returns the coded blocks and an estimate of their bits.
    fn code_plane(&self, fd: &mut FrameDec, plane: usize, intra: bool) -> (PlaneCoding, f64) {
        let h = self.h;
        let tx_sz = if plane > 0 {
            fd.uv_tx_size()
        } else {
            fd.b.tx_size
        };
        let step = 1usize << tx_sz;
        let (base_x, base_y, pw, ph, _) = self.plane_region(fd, plane);
        let (n4w, n4h) = (pw / 4, ph / 4);
        let (sx, sy) = if plane > 0 { (1, 1) } else { (0, 0) };
        let max_x = (h.mi_cols as usize * 8) >> sx;
        let max_y = (h.mi_rows as usize * 8) >> sy;
        let n = 2 + tx_sz as u32;
        let n0 = 1usize << n;
        let q = fd_q(h, plane);
        let dq_denom = if tx_sz == TX_32X32 { 2.0 } else { 1.0 };
        let max_coef = 16450;
        let mut out = Vec::new();
        let mut bits = 0.0;
        let mut block_idx = 0;
        let mut y = 0;
        while y < n4h {
            let mut x = 0;
            while x < n4w {
                let start_x = base_x + 4 * x;
                let start_y = base_y + 4 * y;
                if start_x < max_x && start_y < max_y {
                    if intra {
                        fd.predict_intra(
                            plane,
                            start_x,
                            start_y,
                            fd.b.avail_l || x > 0,
                            fd.b.avail_u || y > 0,
                            x + step < n4w,
                            tx_sz,
                            block_idx,
                        );
                    }
                    let tx_type = fd.tx_type(plane, tx_sz, block_idx);
                    let mut res = vec![0i32; n0 * n0];
                    {
                        let b = &fd.planes[plane];
                        let s = &self.src.planes[plane];
                        let ss = self.src.stride[plane];
                        for i in 0..n0 {
                            for j in 0..n0 {
                                res[i * n0 + j] = s[(start_y + i) * ss + start_x + j] as i32
                                    - b.data[(start_y + i) * b.stride + start_x + j] as i32;
                            }
                        }
                    }
                    let coefs: Vec<i32> = if h.lossless {
                        fdct::forward_wht(&res).to_vec()
                    } else {
                        let d = fdct::forward_2d(&res, n, tx_type);
                        d.iter()
                            .enumerate()
                            .map(|(k, &v)| {
                                let qq = if k == 0 { q.0 } else { q.1 } as f64;
                                let t = v.abs() * dq_denom / qq;
                                let bias = if k == 0 { 0.5 } else { 0.38 };
                                let l = ((t + bias).floor() as i32).min(max_coef);
                                if v < 0.0 { -l } else { l }
                            })
                            .collect()
                    };
                    let scan = scan_for(tx_sz, tx_type);
                    let eob = scan
                        .iter()
                        .rposition(|&p| coefs[p as usize] != 0)
                        .map_or(0, |i| i + 1);
                    if eob > 0 {
                        fd.coefs[..n0 * n0].copy_from_slice(&coefs);
                        fd.reconstruct(plane, start_x, start_y, tx_sz, tx_type, eob);
                        bits += 2.0;
                        for &p in &scan[..eob] {
                            let v = coefs[p as usize];
                            bits += if v == 0 {
                                1.5
                            } else {
                                3.0 + 2.0 * (1.0 + v.unsigned_abs() as f64).log2()
                            };
                        }
                    } else {
                        bits += 1.0;
                    }
                    out.push(Some(TxBlock {
                        coefs,
                        tx_type,
                        eob,
                    }));
                } else {
                    out.push(None);
                }
                block_idx += 1;
                x += step;
            }
            y += step;
        }
        (out, bits)
    }

    /// Codes the luma of the block with intra mode `mode`; returns the
    /// coding and its rate-distortion cost (the reconstruction is left in
    /// the buffer).
    fn try_intra_luma(&self, fd: &mut FrameDec, mode: u8, mode_bits: f64) -> (PlaneCoding, f64) {
        fd.b.y_mode = mode;
        fd.b.sub_modes = [mode; 4];
        let (c, bits) = self.code_plane(fd, 0, true);
        let cost = self.sse(fd, 0) + self.lambda * (bits + mode_bits);
        (c, cost)
    }

    fn try_intra_chroma(
        &self,
        fd: &mut FrameDec,
        mode: u8,
        mode_bits: f64,
    ) -> ([PlaneCoding; 2], f64) {
        fd.b.uv_mode = mode;
        let (c1, b1) = self.code_plane(fd, 1, true);
        let (c2, b2) = self.code_plane(fd, 2, true);
        let cost = self.sse(fd, 1) + self.sse(fd, 2) + self.lambda * (b1 + b2 + mode_bits);
        ([c1, c2], cost)
    }

    fn y_mode_bits(&self, fd: &FrameDec, mode: u8) -> f64 {
        if self.h.frame_is_intra {
            let am = fd.b.above.map_or(DC_PRED, |m| m.sub_modes[2]);
            let lm = fd.b.left.map_or(DC_PRED, |m| m.sub_modes[1]);
            let p = KF_Y_MODE_PROBS[am as usize][lm as usize];
            tree_bits(&INTRA_MODE_TREE, mode, |n| p[n])
        } else {
            let p = fd.probs.y_mode[SIZE_GROUP[fd.b.mi_size as usize] as usize];
            tree_bits(&INTRA_MODE_TREE, mode, |n| p[n])
        }
    }

    fn uv_mode_bits(&self, fd: &FrameDec, y_mode: u8, mode: u8) -> f64 {
        let p = if self.h.frame_is_intra {
            KF_UV_MODE_PROBS[y_mode as usize]
        } else {
            fd.probs.uv_mode[y_mode as usize]
        };
        tree_bits(&INTRA_MODE_TREE, mode, |n| p[n])
    }

    /// The best intra modes for the block and their cost.
    fn best_intra(&self, fd: &mut FrameDec) -> (Choice, f64) {
        fd.b.is_inter = false;
        fd.b.ref_frame = [INTRA_FRAME, NONE];
        let saved = self.save(fd, 0);
        let mut best = (DC_PRED, f64::MAX);
        for mode in 0..10u8 {
            let mb = self.y_mode_bits(fd, mode);
            let (_, cost) = self.try_intra_luma(fd, mode, mb);
            if cost < best.1 {
                best = (mode, cost);
            }
            self.restore(fd, 0, &saved);
        }
        let y_mode = best.0;
        let saved1 = self.save(fd, 1);
        let saved2 = self.save(fd, 2);
        let mut best_uv = (DC_PRED, f64::MAX);
        for mode in 0..10u8 {
            let mb = self.uv_mode_bits(fd, y_mode, mode);
            let (_, cost) = self.try_intra_chroma(fd, mode, mb);
            if cost < best_uv.1 {
                best_uv = (mode, cost);
            }
            self.restore(fd, 1, &saved1);
            self.restore(fd, 2, &saved2);
        }
        (
            Choice::Intra {
                y_mode,
                uv_mode: best_uv.0,
            },
            best.1 + best_uv.1,
        )
    }

    // -----------------------------------------------------------------
    // Inter.

    fn block_sad_full(
        &self,
        fd: &FrameDec,
        mv_row_px: i32,
        mv_col_px: i32,
        w: usize,
        h: usize,
    ) -> u64 {
        let x0 = (fd.b.mi_col * 8) as isize + mv_col_px as isize + PAD as isize;
        let y0 = (fd.b.mi_row * 8) as isize + mv_row_px as isize + PAD as isize;
        let s = &self.src.planes[0];
        let ss = self.src.stride[0];
        let bx = (fd.b.mi_col * 8) as usize;
        let by = (fd.b.mi_row * 8) as usize;
        let mut acc = 0u64;
        for i in 0..h {
            let ry = (y0 + i as isize) as usize;
            let row = &self.padded[ry * self.pad_stride..];
            for j in 0..w {
                let a = row[(x0 + j as isize) as usize] as i32;
                let b = s[(by + i) * ss + bx + j] as i32;
                acc += (a - b).unsigned_abs() as u64;
            }
        }
        acc
    }

    /// SAD of the luma prediction with `mv` (1/8 units) — the decoder's
    /// filter, without its clamps (the search stays inside them).
    fn block_sad_sub(&self, fd: &FrameDec, mv: Mv, w: usize, h: usize, buf: &mut [u16]) -> u64 {
        let r = self.last.unwrap();
        let rp = &r.planes[0];
        let refp = inter::RefPlane {
            data: &rp.data,
            stride: rp.stride,
            last_x: r.width as i32 - 1,
            last_y: r.height as i32 - 1,
        };
        let x = (fd.b.mi_col * 8) as i32 * 16 + mv[1] * 2;
        let y = (fd.b.mi_row * 8) as i32 * 16 + mv[0] * 2;
        inter::predict(&refp, x, y, 16, 16, w, h, EIGHTTAP, 8, &mut buf[..w * h]);
        let s = &self.src.planes[0];
        let ss = self.src.stride[0];
        let bx = (fd.b.mi_col * 8) as usize;
        let by = (fd.b.mi_row * 8) as usize;
        let mut acc = 0u64;
        for i in 0..h {
            for j in 0..w {
                acc += (buf[i * w + j] as i32 - s[(by + i) * ss + bx + j] as i32).unsigned_abs()
                    as u64;
            }
        }
        acc
    }

    /// The range of motion vectors (1/8 units, even) the decoder will not
    /// clamp for the luma of this block.
    fn mv_bounds(&self, fd: &FrameDec) -> (i32, i32, i32, i32) {
        let bh = NUM_8X8_HIGH[fd.b.mi_size as usize] as i32;
        let bw = NUM_8X8_WIDE[fd.b.mi_size as usize] as i32;
        let (r, c) = (fd.b.mi_row as i32, fd.b.mi_col as i32);
        let range = self.cfg.search_range as i32 * 8;
        let top = (-(r * 8 * 8) - (INTERP_EXTEND + bh * 8) * 8 + 8).max(-range);
        let bottom = (((self.h.mi_rows as i32 - bh - r) * 8) * 8 + (INTERP_EXTEND + bh * 8) * 8
            - 16)
            .min(range);
        let left = (-(c * 8 * 8) - (INTERP_EXTEND + bw * 8) * 8 + 8).max(-range);
        let right = (((self.h.mi_cols as i32 - bw - c) * 8) * 8 + (INTERP_EXTEND + bw * 8) * 8
            - 16)
            .min(range);
        // Keep whole-pixel search positions inside the padded reference too.
        let lim = (PAD as i32 - 8) * 8;
        (
            top.max(-lim) & !1,
            bottom.min(lim) & !1,
            left.max(-lim) & !1,
            right.min(lim) & !1,
        )
    }

    fn motion_search(&self, fd: &FrameDec) -> Mv {
        let w = 8 * NUM_8X8_WIDE[fd.b.mi_size as usize] as usize;
        let h = 8 * NUM_8X8_HIGH[fd.b.mi_size as usize] as usize;
        let (top, bottom, left, right) = self.mv_bounds(fd);
        let inside = |m: Mv| m[0] >= top && m[0] <= bottom && m[1] >= left && m[1] <= right;
        let lam = (self.lambda.sqrt() * 1.2).max(1.0);
        let best_mv = fd.b.best_mv[0];
        let cost =
            |sad: u64, m: Mv| sad as f64 + lam * mv_bits([m[0] - best_mv[0], m[1] - best_mv[1]]);
        // Whole pixels.
        let mut best = ([0i32; 2], f64::MAX);
        let mut starts = vec![[0, 0], fd.b.nearest_mv[0], fd.b.near_mv[0]];
        for s in starts.iter_mut() {
            *s = [(s[0] + 4).div_euclid(8) * 8, (s[1] + 4).div_euclid(8) * 8];
        }
        for s in starts {
            if inside(s) {
                let c = cost(self.block_sad_full(fd, s[0] / 8, s[1] / 8, w, h), s);
                if c < best.1 {
                    best = (s, c);
                }
            }
        }
        let mut step = 8 * 8;
        while step >= 8 {
            let mut improved = true;
            while improved {
                improved = false;
                let centre = best.0;
                for d in [
                    [-step, 0],
                    [step, 0],
                    [0, -step],
                    [0, step],
                    [-step, -step],
                    [-step, step],
                    [step, -step],
                    [step, step],
                ] {
                    let m = [centre[0] + d[0], centre[1] + d[1]];
                    if inside(m) {
                        let c = cost(self.block_sad_full(fd, m[0] / 8, m[1] / 8, w, h), m);
                        if c < best.1 {
                            best = (m, c);
                            improved = true;
                        }
                    }
                }
            }
            step /= 2;
        }
        // Half and quarter pixels with the real filter.
        let mut buf = vec![0u16; w * h];
        best.1 = cost(self.block_sad_sub(fd, best.0, w, h, &mut buf), best.0);
        for step in [4, 2] {
            let centre = best.0;
            for d in [
                [-step, 0],
                [step, 0],
                [0, -step],
                [0, step],
                [-step, -step],
                [-step, step],
                [step, -step],
                [step, step],
            ] {
                let m = [centre[0] + d[0], centre[1] + d[1]];
                if inside(m) {
                    let c = cost(self.block_sad_sub(fd, m, w, h, &mut buf), m);
                    if c < best.1 {
                        best = (m, c);
                    }
                }
            }
        }
        best.0
    }

    /// Predicts every plane of the block with `mv` (the decoder's process)
    /// and codes the residual; returns codings and the cost.
    fn try_inter(
        &self,
        fd: &mut FrameDec,
        y_mode: u8,
        mv: Mv,
        mode_bits: f64,
    ) -> ([PlaneCoding; 3], f64) {
        fd.b.is_inter = true;
        fd.b.ref_frame = [LAST_FRAME, NONE];
        fd.b.y_mode = y_mode;
        fd.b.interp_filter = EIGHTTAP;
        fd.b.block_mvs = [[mv; 4], [[0; 2]; 4]];
        let mut codings: [PlaneCoding; 3] = Default::default();
        let mut bits = mode_bits;
        let mut sse = 0.0;
        for (plane, coding) in codings.iter_mut().enumerate() {
            let (x, y, w, h, _) = self.plane_region(fd, plane);
            fd.predict_inter(plane, x, y, w, h, 0)
                .expect("reference present");
            let (c, b) = self.code_plane(fd, plane, false);
            *coding = c;
            bits += b;
            sse += self.sse(fd, plane);
        }
        (codings, sse + self.lambda * bits)
    }

    fn inter_mode_bits(&self, fd: &FrameDec, y_mode: u8) -> f64 {
        let ctx = fd.b.mode_context[LAST_FRAME as usize] as usize;
        let p = fd.probs.inter_mode[ctx];
        tree_bits(&INTER_MODE_TREE, y_mode - NEARESTMV, |n| p[n])
    }

    fn best_inter(&self, fd: &mut FrameDec) -> (Choice, f64) {
        fd.b.ref_frame = [LAST_FRAME, NONE];
        fd.b.is_inter = true;
        fd.find_best_ref_mvs(0);
        let searched = self.motion_search(fd);
        let w = 8 * NUM_8X8_WIDE[fd.b.mi_size as usize] as usize;
        let h = 8 * NUM_8X8_HIGH[fd.b.mi_size as usize] as usize;
        let (top, bottom, left, right) = self.mv_bounds(fd);
        let inside = |m: Mv| m[0] >= top && m[0] <= bottom && m[1] >= left && m[1] <= right;
        let lam = (self.lambda.sqrt() * 1.2).max(1.0);
        let mut buf = vec![0u16; w * h];
        let mut cands = vec![
            (ZEROMV, [0, 0]),
            (NEARESTMV, fd.b.nearest_mv[0]),
            (NEARMV, fd.b.near_mv[0]),
        ];
        if searched != fd.b.nearest_mv[0] && searched != fd.b.near_mv[0] && searched != [0, 0] {
            cands.push((NEWMV, searched));
        }
        let mut best = (ZEROMV, [0, 0], f64::MAX);
        for (mode, mv) in cands {
            if !inside(mv) && mode != NEWMV {
                // The decoder would clamp it; still valid, judge it by the
                // full process below only if nothing else is available.
                continue;
            }
            let mut bits = self.inter_mode_bits(fd, mode);
            if mode == NEWMV {
                bits += mv_bits([mv[0] - fd.b.best_mv[0][0], mv[1] - fd.b.best_mv[0][1]]);
            }
            let c = self.block_sad_sub(fd, mv, w, h, &mut buf) as f64 + lam * bits;
            if c < best.2 {
                best = (mode, mv, c);
            }
        }
        let saved: Vec<Vec<u16>> = (0..3).map(|p| self.save(fd, p)).collect();
        let mut bits = self.inter_mode_bits(fd, best.0);
        if best.0 == NEWMV {
            bits += mv_bits([
                best.1[0] - fd.b.best_mv[0][0],
                best.1[1] - fd.b.best_mv[0][1],
            ]);
        }
        let (_, cost) = self.try_inter(fd, best.0, best.1, bits);
        for (p, s) in saved.iter().enumerate() {
            self.restore(fd, p, s);
        }
        (
            Choice::Inter {
                y_mode: best.0,
                mv: best.1,
            },
            cost,
        )
    }

    // -----------------------------------------------------------------
    // The block: decide, reconstruct, write.

    fn block(&mut self, e: &mut BoolEncoder, fd: &mut FrameDec, r: u32, c: u32, bsize: u8) {
        let avail_u = r > 0;
        let avail_l = c > fd.mi_col_start;
        let tx_size = MAX_TXSIZE_LOOKUP[bsize as usize]
            .min(TX_MODE_TO_BIGGEST_TX_SIZE[self.h.tx_mode as usize]);
        fd.b = Block {
            mi_row: r,
            mi_col: c,
            mi_size: bsize,
            avail_u,
            avail_l,
            above: if avail_u {
                Some(*fd.mi_at(r - 1, c))
            } else {
                None
            },
            left: if avail_l {
                Some(*fd.mi_at(r, c - 1))
            } else {
                None
            },
            tx_size,
            ref_frame: [INTRA_FRAME, NONE],
            ..Block::default()
        };
        let (intra, intra_cost) = self.best_intra(fd);
        let mut choice = intra;
        let mut inter_mode_ctx = None;
        if !self.h.frame_is_intra {
            let (inter, inter_cost) = self.best_inter(fd);
            // Intra costs one more flag in an inter frame either way; the
            // comparison is between the two codings.
            inter_mode_ctx = Some((
                fd.b.mode_context,
                fd.b.nearest_mv,
                fd.b.near_mv,
                fd.b.best_mv,
            ));
            if inter_cost < intra_cost {
                choice = inter;
            }
        }
        // Final coding into the buffer.
        let codings: [PlaneCoding; 3] = match choice {
            Choice::Intra { y_mode, uv_mode } => {
                fd.b.is_inter = false;
                fd.b.ref_frame = [INTRA_FRAME, NONE];
                fd.b.block_mvs = [[[0; 2]; 4]; 2];
                fd.b.interp_filter = 0;
                let (cy, _) = self.try_intra_luma(fd, y_mode, 0.0);
                let ([cu, cv], _) = self.try_intra_chroma(fd, uv_mode, 0.0);
                [cy, cu, cv]
            }
            Choice::Inter { y_mode, mv } => {
                let (m, n, nr, b) = inter_mode_ctx.unwrap();
                fd.b.mode_context = m;
                fd.b.nearest_mv = n;
                fd.b.near_mv = nr;
                fd.b.best_mv = b;
                self.try_inter(fd, y_mode, mv, 0.0).0
            }
        };
        let skip = codings
            .iter()
            .all(|p| p.iter().all(|t| t.as_ref().is_none_or(|t| t.eob == 0)));
        fd.b.skip = skip;
        // Mode info.
        let skip_ctx =
            fd.b.above.map_or(0, |m| m.skip as usize) + fd.b.left.map_or(0, |m| m.skip as usize);
        if self.h.frame_is_intra {
            e.write(skip, fd.probs.skip[skip_ctx]);
            let am = fd.b.above.map_or(DC_PRED, |m| m.sub_modes[2]);
            let lm = fd.b.left.map_or(DC_PRED, |m| m.sub_modes[1]);
            let p = KF_Y_MODE_PROBS[am as usize][lm as usize];
            e.tree(&INTRA_MODE_TREE, fd.b.y_mode, |n| p[n]);
            let p = KF_UV_MODE_PROBS[fd.b.y_mode as usize];
            e.tree(&INTRA_MODE_TREE, fd.b.uv_mode, |n| p[n]);
        } else {
            e.write(skip, fd.probs.skip[skip_ctx]);
            // is_inter, with the decoder's context.
            let left_intra = fd.left_ref()[0] <= INTRA_FRAME;
            let above_intra = fd.above_ref()[0] <= INTRA_FRAME;
            let ctx = if avail_u && avail_l {
                if left_intra && above_intra {
                    3
                } else {
                    (left_intra || above_intra) as usize
                }
            } else if avail_u || avail_l {
                2 * (if avail_u { above_intra } else { left_intra }) as usize
            } else {
                0
            };
            e.write(fd.b.is_inter, fd.probs.is_inter[ctx]);
            if fd.b.is_inter {
                // read_ref_frames(): single reference, LAST.
                let ctx = fd.single_ref_p1_ctx();
                e.write(false, fd.probs.single_ref[ctx][0]);
                let mctx = fd.b.mode_context[LAST_FRAME as usize] as usize;
                let p = fd.probs.inter_mode[mctx];
                e.tree(&INTER_MODE_TREE, fd.b.y_mode - NEARESTMV, |n| p[n]);
                if fd.b.y_mode == NEWMV {
                    let mv = fd.b.block_mvs[0][0];
                    let best = fd.b.best_mv[0];
                    write_mv(e, fd, [mv[0] - best[0], mv[1] - best[1]]);
                }
            } else {
                let p = fd.probs.y_mode[SIZE_GROUP[bsize as usize] as usize];
                e.tree(&INTRA_MODE_TREE, fd.b.y_mode, |n| p[n]);
                let p = fd.probs.uv_mode[fd.b.y_mode as usize];
                e.tree(&INTRA_MODE_TREE, fd.b.uv_mode, |n| p[n]);
            }
        }
        // Residual tokens, plane by plane, with the decoder's contexts.
        for (plane, coding) in codings.iter().enumerate() {
            let tx_sz = if plane > 0 {
                fd.uv_tx_size()
            } else {
                fd.b.tx_size
            };
            let step = 1usize << tx_sz;
            let (base_x, base_y, pw, _, _) = self.plane_region(fd, plane);
            let n4w = pw / 4;
            for (k, tb) in coding.iter().enumerate() {
                let x = (k % n4w.div_ceil(step)) * step;
                let y = (k / n4w.div_ceil(step)) * step;
                let start_x = base_x + 4 * x;
                let start_y = base_y + 4 * y;
                let mut nonzero = false;
                if let Some(tb) = tb
                    && !skip
                {
                    write_tokens(e, fd, &mut self.stats, plane, start_x, start_y, tx_sz, tb);
                    nonzero = tb.eob > 0;
                }
                for i in 0..step {
                    fd.above_nonzero[plane][(start_x >> 2) + i] = nonzero as u8;
                    fd.left_nonzero[plane][(start_y >> 2) + i] = nonzero as u8;
                }
            }
        }
        // What the decoder will remember.
        let b = &fd.b;
        let info = MiInfo {
            mi_size: bsize,
            skip,
            tx_size: b.tx_size,
            y_mode: b.y_mode,
            sub_modes: [b.y_mode; 4],
            segment_id: 0,
            ref_frame: b.ref_frame,
            interp_filter: b.interp_filter,
            mv: b.block_mvs,
        };
        let bh = NUM_8X8_HIGH[bsize as usize] as u32;
        let bw = NUM_8X8_WIDE[bsize as usize] as u32;
        for y in r..(r + bh).min(self.h.mi_rows) {
            for x in c..(c + bw).min(self.h.mi_cols) {
                fd.mi[(y * self.h.mi_cols + x) as usize] = info;
            }
        }
    }
}

/// (dc, ac) quantiser of a plane.
fn fd_q(h: &FrameHeader, plane: usize) -> (i32, i32) {
    let q = h.base_q_idx;
    let (dcd, acd) = if plane == 0 {
        (h.delta_q_y_dc, 0)
    } else {
        (h.delta_q_uv_dc, h.delta_q_uv_ac)
    };
    (
        DC_QLOOKUP[0][(q + dcd).clamp(0, 255) as usize],
        AC_QLOOKUP[0][(q + acd).clamp(0, 255) as usize],
    )
}

fn scan_for(tx_sz: u8, tx_type: u8) -> &'static [u16] {
    match (tx_sz, tx_type) {
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
    }
}

/// The inverse of tokens() (6.4.24): the same contexts, writing.
#[allow(clippy::too_many_arguments)]
fn write_tokens(
    e: &mut BoolEncoder,
    fd: &mut FrameDec,
    stats: &mut CoefStats,
    plane: usize,
    start_x: usize,
    start_y: usize,
    tx_sz: u8,
    tb: &TxBlock,
) {
    let scan = scan_for(tx_sz, tb.tx_type);
    let seg_eob = 16usize << (tx_sz << 1);
    let ref_type = fd.b.is_inter as usize;
    let ptype = (plane > 0) as usize;
    let txs = tx_sz as usize;
    let (sx, sy) = if plane > 0 { (1, 1) } else { (0, 0) };
    let max_x4 = ((2 * fd.mi_cols) >> sx) as usize;
    let max_y4 = ((2 * fd.mi_rows) >> sy) as usize;
    let x4 = start_x >> 2;
    let y4 = start_y >> 2;
    let mut above = 0u8;
    let mut left = 0u8;
    for i in 0..(1usize << tx_sz) {
        if x4 + i < max_x4 {
            above |= fd.above_nonzero[plane][x4 + i];
        }
        if y4 + i < max_y4 {
            left |= fd.left_nonzero[plane][y4 + i];
        }
    }
    let mut ctx = (above + left) as usize;
    let n = 4usize << tx_sz;
    let log2n = 2 + tx_sz as u32;
    let mut check_eob = true;
    let mut c = 0;
    while c < seg_eob {
        let pos = scan[c] as usize;
        let band = if tx_sz == TX_4X4 {
            COEFBAND_4X4[c]
        } else {
            COEFBAND_8X8PLUS[c]
        } as usize;
        if c > 0 {
            let i = pos >> log2n;
            let j = pos & (n - 1);
            let (a, b) = if i > 0 && j > 0 {
                let a = (i - 1) * n + j;
                let a2 = i * n + j - 1;
                match tb.tx_type {
                    DCT_ADST => (a, a),
                    ADST_DCT => (a2, a2),
                    _ => (a, a2),
                }
            } else if i > 0 {
                ((i - 1) * n + j, (i - 1) * n + j)
            } else {
                (j - 1, j - 1)
            };
            ctx = (1 + fd.token_cache[a] as usize + fd.token_cache[b] as usize) >> 1;
        }
        let probs = fd.probs.coef[txs][ptype][ref_type][band][ctx];
        let st = &mut stats[txs][ptype][ref_type][band][ctx];
        if check_eob {
            let more = c < tb.eob;
            st[0][more as usize] += 1;
            e.write(more, probs[0]);
            if !more {
                break;
            }
        }
        let v = tb.coefs[pos];
        let mag = v.unsigned_abs() as i32;
        let token = match mag {
            0..=4 => mag as u8,
            5..=6 => 5,
            7..=10 => 6,
            11..=18 => 7,
            19..=34 => 8,
            35..=66 => 9,
            _ => 10,
        };
        st[1][(token != ZERO_TOKEN) as usize] += 1;
        if token != ZERO_TOKEN {
            st[2][(token > 1) as usize] += 1;
        }
        e.tree(&TOKEN_TREE, token, |node| {
            pareto(node, probs[(1 + node).min(2)])
        });
        fd.token_cache[pos] = ENERGY_CLASS[token as usize];
        if token == ZERO_TOKEN {
            check_eob = false;
        } else {
            if token >= 5 {
                let [cat, num_extra, base] = EXTRA_BITS[token as usize];
                let extra = mag - base;
                let probs = CAT_PROBS[cat as usize];
                for k in 0..num_extra {
                    e.write((extra >> (num_extra - 1 - k)) & 1 != 0, probs[k as usize]);
                }
            }
            e.write(v < 0, 128);
            check_eob = true;
        }
        c += 1;
    }
}

/// read_mv's inverse for a difference `d` (both components even: no high
/// precision).
fn write_mv(e: &mut BoolEncoder, fd: &FrameDec, d: Mv) {
    let joint = (((d[0] != 0) as u8) << 1) | (d[1] != 0) as u8;
    let p = fd.probs.mv_joint;
    e.tree(&MV_JOINT_TREE, joint, |n| p[n]);
    for comp in 0..2 {
        let v = d[comp];
        if v == 0 {
            continue;
        }
        e.write(v < 0, fd.probs.mv_sign[comp]);
        let z = v.unsigned_abs() - 1;
        let class = if z < 16 {
            0
        } else {
            (31 - (z >> 3).leading_zeros()) as u8
        };
        let pc = fd.probs.mv_class[comp];
        e.tree(&MV_CLASS_TREE, class, |n| pc[n]);
        if class == 0 {
            let c0 = (z >> 3) & 1;
            let fr = (z >> 1) & 3;
            e.write(c0 != 0, fd.probs.mv_class0_bit[comp]);
            let pf = fd.probs.mv_class0_fr[comp][c0 as usize];
            e.tree(&MV_FR_TREE, fr as u8, |n| pf[n]);
        } else {
            let off = z - (2 << (class + 2));
            let dd = off >> 3;
            let fr = (off >> 1) & 3;
            for i in 0..class as usize {
                e.write((dd >> i) & 1 != 0, fd.probs.mv_bits[comp][i]);
            }
            let pf = fd.probs.mv_fr[comp];
            e.tree(&MV_FR_TREE, fr as u8, |n| pf[n]);
        }
        // hp is implied 1 (allow_high_precision_mv is 0): z is odd.
        debug_assert!(z & 1 == 1, "odd motion vector difference");
    }
}
