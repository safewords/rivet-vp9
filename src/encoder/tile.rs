//! Block decisions and the tile syntax writer.
//!
//! The decoder's `FrameDec` is the reconstruction engine: its prediction,
//! motion vector prediction and reconstruct functions are called exactly as
//! the residual syntax calls them, on its picture buffers, so the encoder's
//! picture is the decoder's picture.
//!
//! Decisions are made by coding: a candidate (a partition of a region, a
//! transform size, intra against inter) is predicted, transformed,
//! quantised and reconstructed for real, and its syntax written to a
//! [`BitCounter`] — the exact rate under the frame's probabilities — then
//! judged by squared error plus `lambda` times that rate. The region's state
//! (samples, mode info, the nonzero and partition contexts) is saved before
//! a trial and put back after it. A superblock's search ends with its state
//! put back and the winning decisions replayed into the boolean encoder.

// Loops index arrays the way the specification's formulas do.
#![allow(clippy::needless_range_loop)]

use crate::bool_coder::{BitCounter, BoolEncoder, Sink};
use crate::consts::*;
use crate::decoder::block::{Block, Mv, pareto};
use crate::decoder::{FrameDec, MiInfo, RefFrame};
use crate::dsp::{inter, pixel};
use crate::frame::Frame;
use crate::header::FrameHeader;
use crate::tables::*;

use super::Config;
use super::{fdct, scratch};

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
            let (pw, ph) = if p == 0 {
                (w, ht)
            } else {
                (w >> h.subsampling_x, ht >> h.subsampling_y)
            };
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
struct TxBlock<'a> {
    coefs: &'a [i32],
    tx_type: u8,
    eob: usize,
}

/// The transform blocks of one plane of a block, in syntax order; `None`
/// for those outside the frame. Their coefficients are kept one after the
/// other. The buffers come from and go back to the [`scratch`] pools.
struct PlaneCoding {
    /// (offset in `coefs`, size, tx_type, eob).
    blocks: Vec<Option<(usize, usize, u8, usize)>>,
    coefs: Vec<i32>,
}

impl Default for PlaneCoding {
    fn default() -> Self {
        PlaneCoding {
            blocks: scratch::take_blocks(),
            coefs: scratch::take_i32(),
        }
    }
}

impl Drop for PlaneCoding {
    fn drop(&mut self) {
        scratch::give_blocks(std::mem::take(&mut self.blocks));
        scratch::give_i32(std::mem::take(&mut self.coefs));
    }
}

impl PlaneCoding {
    fn iter(&self) -> impl Iterator<Item = Option<TxBlock<'_>>> {
        self.blocks.iter().map(|b| {
            b.map(|(off, len, tx_type, eob)| TxBlock {
                coefs: &self.coefs[off..off + len],
                tx_type,
                eob,
            })
        })
    }
}

/// What a block is coded with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Choice {
    pub is_inter: bool,
    /// Intra: the luma mode; inter: NEARESTMV, NEARMV, ZEROMV or NEWMV.
    pub y_mode: u8,
    pub uv_mode: u8,
    /// Inter: the motion vector (1/8 sample).
    pub mv: Mv,
    /// Inter: LAST_FRAME or GOLDEN_FRAME.
    pub ref_frame: i8,
    pub tx_size: u8,
}

/// One decision, in the order the syntax needs them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Decision {
    Partition(u8),
    Block(Choice),
}

/// How hard the encoder searches (from [`Config::speed`]).
#[derive(Clone, Copy, Debug)]
struct Effort {
    /// Search the partition (else the fixed partition of
    /// [`Config::block_size`]).
    partition: bool,
    /// Try HORZ and VERT where the frame edge does not force them.
    rect: bool,
    /// Transform sizes to try below the largest.
    tx_depth: u8,
}

/// The state of a region of the frame, saved around a trial.
struct Snap {
    r: u32,
    c: u32,
    bw8: u32,
    bh8: u32,
    planes: [Vec<u16>; 3],
    mi: Vec<MiInfo>,
    above_nz: [Vec<u8>; 3],
    left_nz: [Vec<u8>; 3],
    above_part: Vec<u8>,
    left_part: Vec<u8>,
}

impl Drop for Snap {
    fn drop(&mut self) {
        for v in &mut self.planes {
            scratch::give_u16(std::mem::take(v));
        }
        scratch::give_mi(std::mem::take(&mut self.mi));
        for v in self.above_nz.iter_mut().chain(self.left_nz.iter_mut()) {
            scratch::give_u8(std::mem::take(v));
        }
        scratch::give_u8(std::mem::take(&mut self.above_part));
        scratch::give_u8(std::mem::take(&mut self.left_part));
    }
}

/// A copy of `src` in a buffer from the [`scratch`] pool.
fn pooled_u8(src: &[u8]) -> Vec<u8> {
    let mut v = scratch::take_u8();
    v.extend_from_slice(src);
    v
}

pub(crate) struct TileEncoder<'a> {
    cfg: &'a Config,
    h: &'a FrameHeader,
    src: &'a Source,
    /// The references searched (LAST, then GOLDEN when it is another
    /// picture): the frame, and its luma padded for whole-pixel search.
    refs: &'a [SearchRef<'a>],
    pad_stride: usize,
    /// Rate-distortion multiplier (squared error per bit).
    lambda: f64,
    effort: Effort,
    /// How often each of the first three coefficient probabilities of
    /// every context saw a 0 and a 1: what forward updates are judged by.
    pub(crate) stats: Box<CoefStats>,
    /// Whether coding adds to `stats` (not in trials).
    record: bool,
    /// Every decision, in coding order.
    pub(crate) decisions: Vec<Decision>,
    /// Decisions to replay instead of searching, and the next one.
    pub(crate) replay: Option<(Vec<Decision>, usize)>,
    /// The kernels' instruction set, and inter prediction's memory.
    level: crate::dsp::Level,
    scratch: std::cell::RefCell<inter::Scratch>,
}

/// A reference frame the motion search looks in: the frame, and its luma
/// padded by [`PAD`] for whole-pixel search (row stride `stride`).
pub(crate) struct SearchRef<'a> {
    ref_frame: i8,
    frame: &'a RefFrame,
    padded: Vec<u16>,
    stride: usize,
}

impl<'a> SearchRef<'a> {
    /// `r`, padded: built once per frame, shared by every tile.
    pub(crate) fn new(h: &FrameHeader, ref_frame: i8, r: &'a RefFrame) -> Self {
        let ps = (h.sb64_cols * 64) as usize + 2 * PAD;
        let rows = (h.sb64_rows * 64) as usize + 2 * PAD;
        let w = r.width as usize;
        let ht = r.height as usize;
        let rp = &r.planes[0];
        let mut v = Vec::with_capacity(ps * rows);
        for y in 0..rows {
            let sy = (y as isize - PAD as isize).clamp(0, ht as isize - 1) as usize;
            let row = &rp.data[sy * rp.stride..sy * rp.stride + w];
            v.resize(v.len() + PAD, row[0]);
            v.extend_from_slice(row);
            let right = ps - PAD - w;
            v.resize(v.len() + right, row[w - 1]);
        }
        SearchRef {
            ref_frame,
            frame: r,
            padded: v,
            stride: ps,
        }
    }
}

/// `[txSz][plane > 0][is_inter][band][ctx][node][bit]`.
pub(crate) type CoefStats = [[[[[[[u32; 2]; 3]; 6]; 6]; 2]; 2]; 4];

const PAD: usize = 96;

fn tree_bits(tree: &[i8], value: u8, prob: impl FnMut(usize) -> u8) -> f64 {
    let mut c = BitCounter::default();
    c.tree(tree, value, prob);
    c.bits
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
        refs: &'a [SearchRef<'a>],
    ) -> Self {
        // The quantiser step of the frame's bit depth: at 10 and 12 bits it
        // is about 4 and 16 times the 8-bit step, so the multiplier (squared
        // error per bit) scales by 2^(2(BitDepth - 8)) with it, as the
        // squared error itself does.
        let q = AC_QLOOKUP[bd_index(h)][h.base_q_idx as usize] as f64 / 8.0;
        let lambda = 0.12 * q * q;
        let effort = match cfg.speed {
            0 => Effort {
                partition: true,
                rect: true,
                tx_depth: 3,
            },
            1 => Effort {
                partition: true,
                rect: false,
                tx_depth: 1,
            },
            _ => Effort {
                partition: false,
                rect: false,
                tx_depth: 0,
            },
        };
        let mut te = TileEncoder {
            cfg,
            h,
            src,
            refs,
            pad_stride: 0,
            lambda,
            effort,
            stats: Box::new([[[[[[[0; 2]; 3]; 6]; 6]; 2]; 2]; 4]),
            record: true,
            decisions: Vec::new(),
            replay: None,
            level: crate::dsp::level(),
            scratch: std::cell::RefCell::new(inter::Scratch::new()),
        };
        te.pad_stride = refs.first().map_or(0, |r| r.stride);
        te
    }

    /// (subsampling_x, subsampling_y) of `plane`.
    fn ss(&self, plane: usize) -> (usize, usize) {
        if plane > 0 {
            (self.h.subsampling_x as usize, self.h.subsampling_y as usize)
        } else {
            (0, 0)
        }
    }

    /// Whether a block of `bsize` has a chroma block size the specification
    /// allows with the frame's subsampling (4:2:2 forbids blocks twice as
    /// tall as wide, 4:4:0 twice as wide as tall: their chroma would be 4:1).
    fn chroma_ok(&self, bsize: u8) -> bool {
        bsize < BLOCK_8X8
            || SS_SIZE_LOOKUP[bsize as usize][self.h.subsampling_x as usize]
                [self.h.subsampling_y as usize]
                != BLOCK_INVALID
    }

    /// Codes tile column `tile_col` (every tile row: the encoder codes one)
    /// into `fd`, a decoder over that column's samples and mode info.
    /// Returns the tile's data.
    pub(crate) fn encode_column(&mut self, fd: &mut FrameDec, tile_col: u32) -> Vec<u8> {
        let h = self.h;
        for p in fd.above_nonzero.iter_mut() {
            p.iter_mut().for_each(|v| *v = 0);
        }
        fd.above_partition.iter_mut().for_each(|v| *v = 0);
        fd.above_seg_pred.iter_mut().for_each(|v| *v = 0);
        let search = self.replay.is_none() && self.effort.partition;
        let (start, end) = column_bounds(h, tile_col);
        fd.mi_row_start = 0;
        fd.mi_row_end = h.mi_rows;
        fd.mi_col_start = start;
        fd.mi_col_end = end;
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
                if search {
                    // Search, put the superblock back as it was, and
                    // code the winner.
                    let pre = self.snapshot(fd, r, c, 8, 8);
                    self.record = false;
                    let (_, decs) = self.search(fd, r, c, BLOCK_64X64);
                    self.record = true;
                    self.restore(fd, &pre);
                    self.replay = Some((decs, 0));
                    self.partition(&mut e, fd, r, c, BLOCK_64X64, None);
                    self.replay = None;
                } else {
                    self.partition(&mut e, fd, r, c, BLOCK_64X64, None);
                }
                c += 8;
            }
            r += 8;
        }
        e.finish()
    }

    fn target_size(&self) -> u8 {
        match self.cfg.block_size {
            8 => BLOCK_8X8,
            16 => BLOCK_16X16,
            32 => BLOCK_32X32,
            _ => BLOCK_64X64,
        }
    }

    fn next_replay(&mut self) -> Option<Decision> {
        self.replay.as_mut().map(|(v, i)| {
            *i += 1;
            v[*i - 1]
        })
    }

    // -----------------------------------------------------------------
    // Saving and restoring a region.

    /// The region of plane `plane` covered by `bw8` x `bh8` 8x8 units at
    /// (`r`, `c`): (x, y, w, h).
    fn region(&self, plane: usize, r: u32, c: u32, bw8: u32, bh8: u32) -> [usize; 4] {
        let (sx, sy) = self.ss(plane);
        [
            ((c * 8) as usize) >> sx,
            ((r * 8) as usize) >> sy,
            ((bw8 * 8) as usize) >> sx,
            ((bh8 * 8) as usize) >> sy,
        ]
    }

    fn snapshot(&self, fd: &FrameDec, r: u32, c: u32, bw8: u32, bh8: u32) -> Snap {
        let mut planes: [Vec<u16>; 3] = Default::default();
        let mut above_nz: [Vec<u8>; 3] = Default::default();
        let mut left_nz: [Vec<u8>; 3] = Default::default();
        for p in 0..3 {
            let [x, y, w, h] = self.region(p, r, c, bw8, bh8);
            let b = &fd.planes[p];
            let mut v = scratch::take_u16();
            for i in 0..h {
                let at = b.at(x, y + i);
                v.extend_from_slice(&b.data[at..at + w]);
            }
            planes[p] = v;
            above_nz[p] = pooled_u8(&fd.above_nonzero[p][x >> 2..(x + w).div_ceil(4)]);
            left_nz[p] = pooled_u8(&fd.left_nonzero[p][y >> 2..(y + h).div_ceil(4)]);
        }
        let mut mi = scratch::take_mi();
        for y in r..(r + bh8).min(self.h.mi_rows) {
            for x in c..(c + bw8).min(self.h.mi_cols) {
                mi.push(*fd.mi_at(y, x));
            }
        }
        Snap {
            r,
            c,
            bw8,
            bh8,
            planes,
            mi,
            above_nz,
            left_nz,
            above_part: pooled_u8(&fd.above_partition[c as usize..(c + bw8) as usize]),
            left_part: pooled_u8(&fd.left_partition[r as usize..(r + bh8) as usize]),
        }
    }

    fn restore(&self, fd: &mut FrameDec, s: &Snap) {
        for p in 0..3 {
            let [x, y, w, h] = self.region(p, s.r, s.c, s.bw8, s.bh8);
            let b = &mut fd.planes[p];
            for i in 0..h {
                let at = b.at(x, y + i);
                b.data[at..at + w].copy_from_slice(&s.planes[p][i * w..(i + 1) * w]);
            }
            fd.above_nonzero[p][x >> 2..(x + w).div_ceil(4)].copy_from_slice(&s.above_nz[p]);
            fd.left_nonzero[p][y >> 2..(y + h).div_ceil(4)].copy_from_slice(&s.left_nz[p]);
        }
        let mut k = 0;
        for y in s.r..(s.r + s.bh8).min(self.h.mi_rows) {
            for x in s.c..(s.c + s.bw8).min(self.h.mi_cols) {
                let at = fd.mi_idx(y, x);
                fd.mi[at] = s.mi[k];
                k += 1;
            }
        }
        fd.above_partition[s.c as usize..(s.c + s.bw8) as usize].copy_from_slice(&s.above_part);
        fd.left_partition[s.r as usize..(s.r + s.bh8) as usize].copy_from_slice(&s.left_part);
    }

    /// Squared error of the reconstruction of a region against the
    /// source, over the part inside the frame.
    fn region_sse(&self, fd: &FrameDec, r: u32, c: u32, bw8: u32, bh8: u32) -> f64 {
        (0..3)
            .map(|p| {
                let [x, y, w, h] = self.region(p, r, c, bw8, bh8);
                self.sse_rect(fd, p, x, y, w, h)
            })
            .sum()
    }

    fn sse_rect(&self, fd: &FrameDec, plane: usize, x: usize, y: usize, w: usize, h: usize) -> f64 {
        let (sx, sy) = self.ss(plane);
        let max_x = ((self.h.width as usize + sx) >> sx).min(x + w);
        let max_y = ((self.h.height as usize + sy) >> sy).min(y + h);
        if max_x <= x || max_y <= y {
            return 0.0;
        }
        let b = &fd.planes[plane];
        let ss = self.src.stride[plane];
        pixel::sse(
            self.level,
            &b.data[b.at(x, y)..],
            b.stride,
            &self.src.planes[plane][y * ss + x..],
            ss,
            max_x - x,
            max_y - y,
        ) as f64
    }

    // -----------------------------------------------------------------
    // Partitions.

    /// The partitions allowed for a block of `bsize` at (`r`, `c`) that
    /// the search tries, NONE first and SPLIT last.
    fn candidates(&self, r: u32, c: u32, bsize: u8) -> Vec<u8> {
        if bsize == BLOCK_8X8 {
            return vec![PARTITION_NONE];
        }
        let half = NUM_8X8_WIDE[bsize as usize] as u32 >> 1;
        let has_rows = (r + half) < self.h.mi_rows;
        let has_cols = (c + half) < self.h.mi_cols;
        let ok = |p: u8| self.chroma_ok(SUBSIZE_LOOKUP[p as usize][bsize as usize]);
        let mut v = Vec::with_capacity(4);
        if has_rows && has_cols {
            v.push(PARTITION_NONE);
            if self.effort.rect {
                v.extend(
                    [PARTITION_HORZ, PARTITION_VERT]
                        .into_iter()
                        .filter(|&p| ok(p)),
                );
            }
        } else if has_cols {
            if ok(PARTITION_HORZ) {
                v.push(PARTITION_HORZ);
            }
        } else if has_rows && ok(PARTITION_VERT) {
            v.push(PARTITION_VERT);
        }
        v.push(PARTITION_SPLIT);
        v
    }

    /// Searches the partition of the block of `bsize` at (`r`, `c`). Leaves
    /// the region coded with the best one; returns its cost and decisions.
    fn search(&mut self, fd: &mut FrameDec, r: u32, c: u32, bsize: u8) -> (f64, Vec<Decision>) {
        if r >= self.h.mi_rows || c >= self.h.mi_cols {
            return (0.0, Vec::new());
        }
        let n8 = NUM_8X8_WIDE[bsize as usize] as u32;
        let cands = self.candidates(r, c, bsize);
        let pre = (cands.len() > 1).then(|| self.snapshot(fd, r, c, n8, n8));
        let mut best: Option<(f64, Vec<Decision>, Option<Snap>)> = None;
        for (i, &p) in cands.iter().enumerate() {
            let last = i + 1 == cands.len();
            let (cost, decs) = if p == PARTITION_SPLIT {
                let mut ctr = BitCounter::default();
                self.write_partition_symbol(&mut ctr, fd, r, c, bsize, p);
                let mut cost = self.lambda * ctr.bits;
                let mut decs = vec![Decision::Partition(p)];
                let sub = SUBSIZE_LOOKUP[p as usize][bsize as usize];
                let half = n8 >> 1;
                for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
                    let (cq, dq) = self.search(fd, r + dr, c + dc, sub);
                    cost += cq;
                    decs.extend(dq);
                    if best.as_ref().is_some_and(|b| cost >= b.0) {
                        break; // already worse
                    }
                }
                (cost, decs)
            } else {
                let mut ctr = BitCounter::default();
                let start = self.decisions.len();
                self.partition(&mut ctr, fd, r, c, bsize, Some(p));
                let decs: Vec<Decision> = self.decisions.drain(start..).collect();
                (
                    self.region_sse(fd, r, c, n8, n8) + self.lambda * ctr.bits,
                    decs,
                )
            };
            if best.as_ref().is_none_or(|b| cost < b.0) {
                // The last candidate's state stays in place: no snapshot.
                let post = (!last).then(|| self.snapshot(fd, r, c, n8, n8));
                best = Some((cost, decs, post));
            }
            if !last && let Some(pre) = &pre {
                self.restore(fd, pre);
            }
        }
        let (cost, decs, post) = best.expect("a candidate at least");
        if let Some(post) = post {
            self.restore(fd, &post);
        }
        (cost, decs)
    }

    /// The partition symbol, with the decoder's context.
    fn write_partition_symbol<S: Sink>(
        &self,
        e: &mut S,
        fd: &FrameDec,
        r: u32,
        c: u32,
        bsize: u8,
        partition: u8,
    ) {
        let h = self.h;
        let num8x8 = NUM_8X8_WIDE[bsize as usize] as u32;
        let half = num8x8 >> 1;
        let has_rows = (r + half) < h.mi_rows;
        let has_cols = (c + half) < h.mi_cols;
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
    }

    /// Codes the block of `bsize` at (`r`, `c`): with partition `force`,
    /// else the replayed one, else the fixed partition's.
    fn partition<S: Sink>(
        &mut self,
        e: &mut S,
        fd: &mut FrameDec,
        r: u32,
        c: u32,
        bsize: u8,
        force: Option<u8>,
    ) {
        let h = self.h;
        if r >= h.mi_rows || c >= h.mi_cols {
            return;
        }
        let num8x8 = NUM_8X8_WIDE[bsize as usize] as u32;
        let half = num8x8 >> 1;
        let has_rows = (r + half) < h.mi_rows;
        let has_cols = (c + half) < h.mi_cols;
        let replayed = if force.is_none() {
            match self.next_replay() {
                Some(Decision::Partition(p)) => Some(p),
                None => None,
                Some(d) => unreachable!("replay out of step: {d:?}"),
            }
        } else {
            None
        };
        let partition = force.or(replayed).unwrap_or_else(|| {
            let mut p = if bsize > self.target_size() {
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
            // At the frame edge, a forced HORZ / VERT whose halves would
            // have a forbidden chroma size (4:4:0 / 4:2:2) is split instead.
            if !self.chroma_ok(SUBSIZE_LOOKUP[p as usize][bsize as usize]) {
                p = PARTITION_SPLIT;
            }
            p
        });
        self.decisions.push(Decision::Partition(partition));
        self.write_partition_symbol(e, fd, r, c, bsize, partition);
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
            self.partition(e, fd, r, c, subsize, None);
            self.partition(e, fd, r, c + half, subsize, None);
            self.partition(e, fd, r + half, c, subsize, None);
            self.partition(e, fd, r + half, c + half, subsize, None);
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
        let (sx, sy) = self.ss(plane);
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
        let mut v = scratch::take_u16();
        for i in 0..h {
            let at = b.at(x, y + i);
            v.extend_from_slice(&b.data[at..at + w]);
        }
        v
    }

    fn restore_plane(&self, fd: &mut FrameDec, plane: usize, saved: &[u16]) {
        let (x, y, w, h, _) = self.plane_region(fd, plane);
        let b = &mut fd.planes[plane];
        for i in 0..h {
            let at = b.at(x, y + i);
            b.data[at..at + w].copy_from_slice(&saved[i * w..(i + 1) * w]);
        }
    }

    /// Squared error of the reconstruction against the source over the
    /// visible part of the current block's plane.
    fn sse(&self, fd: &FrameDec, plane: usize) -> f64 {
        let (x, y, w, h, _) = self.plane_region(fd, plane);
        self.sse_rect(fd, plane, x, y, w, h)
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
        let (sx, sy) = self.ss(plane);
        let max_x = (h.mi_cols as usize * 8) >> sx;
        let max_y = (h.mi_rows as usize * 8) >> sy;
        let n = 2 + tx_sz as u32;
        let n0 = 1usize << n;
        let q = fd_q(h, plane);
        let dq_denom = if tx_sz == TX_32X32 { 2 } else { 1 };
        // The largest magnitude a token can carry: category 6 has 14 extra
        // bits, plus BitDepth - 8 high bits above 8 bits.
        let max_coef = 67 + (1i32 << (14 + h.bit_depth - 8)) - 1;
        let mut out = PlaneCoding::default();
        let mut bits = 0.0;
        let mut block_idx = 0;
        let mut res = [0i16; 32 * 32];
        let mut d = [0i32; 32 * 32];
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
                    {
                        let b = &fd.planes[plane];
                        let s = &self.src.planes[plane];
                        let ss = self.src.stride[plane];
                        for i in 0..n0 {
                            let sr = &s[(start_y + i) * ss + start_x..][..n0];
                            let br = &b.data[b.at(start_x, start_y + i)..][..n0];
                            for ((r, &a), &p) in res[i * n0..i * n0 + n0].iter_mut().zip(sr).zip(br)
                            {
                                *r = (a as i32 - p as i32) as i16;
                            }
                        }
                    }
                    let off = out.coefs.len();
                    out.coefs.resize(off + n0 * n0, 0);
                    let coefs = &mut out.coefs[off..];
                    if h.lossless {
                        coefs.copy_from_slice(&fdct::forward_wht(&res));
                    } else {
                        let k = fdct::forward(self.level, &res, n, tx_type, h.bit_depth, &mut d);
                        fdct::quantize(self.level, &d[..n0 * n0], k, q, dq_denom, max_coef, coefs);
                    }
                    let scan = scan_for(tx_sz, tx_type);
                    let eob = eob_of(scan, coefs);
                    if eob > 0 {
                        fd.coefs[..n0 * n0].copy_from_slice(coefs);
                        fd.reconstruct(plane, start_x, start_y, tx_sz, tx_type, eob);
                        bits += 2.0 + bits_of(&scan[..eob], coefs);
                    } else {
                        bits += 1.0;
                    }
                    out.blocks.push(Some((off, n0 * n0, tx_type, eob)));
                } else {
                    out.blocks.push(None);
                }
                block_idx += 1;
                x += step;
            }
            y += step;
        }
        (out, bits)
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

    /// The best intra modes of the block for each transform size in
    /// `tx_set` (the first the largest), judged on estimated bits: the luma
    /// mode per size, the chroma mode once (at the largest).
    fn intra_candidates(&self, fd: &mut FrameDec, tx_set: &[u8]) -> Vec<Choice> {
        fd.b.is_inter = false;
        fd.b.ref_frame = [INTRA_FRAME, NONE];
        let saved: Vec<Vec<u16>> = (0..3).map(|p| self.save(fd, p)).collect();
        let mut y_modes = Vec::with_capacity(tx_set.len());
        for &t in tx_set {
            fd.b.tx_size = t;
            let mut best = (DC_PRED, f64::MAX);
            for mode in 0..10u8 {
                fd.b.y_mode = mode;
                fd.b.sub_modes = [mode; 4];
                let (_, bits) = self.code_plane(fd, 0, true);
                let cost = self.sse(fd, 0) + self.lambda * (bits + self.y_mode_bits(fd, mode));
                if cost < best.1 {
                    best = (mode, cost);
                }
                self.restore_plane(fd, 0, &saved[0]);
            }
            y_modes.push(best.0);
        }
        fd.b.tx_size = tx_set[0];
        fd.b.y_mode = y_modes[0];
        let mut best_uv = (DC_PRED, f64::MAX);
        for mode in 0..10u8 {
            fd.b.uv_mode = mode;
            let (_, b1) = self.code_plane(fd, 1, true);
            let (_, b2) = self.code_plane(fd, 2, true);
            let mb = self.uv_mode_bits(fd, y_modes[0], mode);
            let cost = self.sse(fd, 1) + self.sse(fd, 2) + self.lambda * (b1 + b2 + mb);
            if cost < best_uv.1 {
                best_uv = (mode, cost);
            }
            self.restore_plane(fd, 1, &saved[1]);
            self.restore_plane(fd, 2, &saved[2]);
        }
        for v in saved {
            scratch::give_u16(v);
        }
        tx_set
            .iter()
            .zip(y_modes)
            .map(|(&t, y_mode)| Choice {
                is_inter: false,
                y_mode,
                uv_mode: best_uv.0,
                mv: [0, 0],
                ref_frame: INTRA_FRAME,
                tx_size: t,
            })
            .collect()
    }

    // -----------------------------------------------------------------
    // Inter.

    fn block_sad_full(
        &self,
        fd: &FrameDec,
        ri: usize,
        mv_row_px: i32,
        mv_col_px: i32,
        w: usize,
        h: usize,
    ) -> u64 {
        let x0 = (fd.b.mi_col * 8) as isize + mv_col_px as isize + PAD as isize;
        let y0 = (fd.b.mi_row * 8) as isize + mv_row_px as isize + PAD as isize;
        let ss = self.src.stride[0];
        let bx = (fd.b.mi_col * 8) as usize;
        let by = (fd.b.mi_row * 8) as usize;
        pixel::sad(
            self.level,
            &self.refs[ri].padded[y0 as usize * self.pad_stride + x0 as usize..],
            self.pad_stride,
            &self.src.planes[0][by * ss + bx..],
            ss,
            w,
            h,
        )
    }

    /// SAD of the luma prediction with `mv` (1/8 units) — the decoder's
    /// filter, without its clamps (the search stays inside them).
    #[allow(clippy::too_many_arguments)]
    fn block_sad_sub(
        &self,
        fd: &FrameDec,
        ri: usize,
        mv: Mv,
        w: usize,
        h: usize,
        buf: &mut [u16],
    ) -> u64 {
        let r = self.refs[ri].frame;
        let rp = &r.planes[0];
        let refp = inter::RefPlane {
            data: &rp.data,
            stride: rp.stride,
            last_x: r.width as i32 - 1,
            last_y: r.height as i32 - 1,
        };
        let x = (fd.b.mi_col * 8) as i32 * 16 + mv[1] * 2;
        let y = (fd.b.mi_row * 8) as i32 * 16 + mv[0] * 2;
        inter::predict(
            self.level,
            &mut self.scratch.borrow_mut(),
            &refp,
            x,
            y,
            16,
            16,
            w,
            h,
            EIGHTTAP,
            self.h.bit_depth,
            &mut buf[..w * h],
            w,
        );
        let ss = self.src.stride[0];
        let bx = (fd.b.mi_col * 8) as usize;
        let by = (fd.b.mi_row * 8) as usize;
        pixel::sad(
            self.level,
            &buf[..w * h],
            w,
            &self.src.planes[0][by * ss + bx..],
            ss,
            w,
            h,
        )
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

    fn motion_search(&self, fd: &FrameDec, ri: usize) -> Mv {
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
                let c = cost(self.block_sad_full(fd, ri, s[0] / 8, s[1] / 8, w, h), s);
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
                        let c = cost(self.block_sad_full(fd, ri, m[0] / 8, m[1] / 8, w, h), m);
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
        best.1 = cost(self.block_sad_sub(fd, ri, best.0, w, h, &mut buf), best.0);
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
                    let c = cost(self.block_sad_sub(fd, ri, m, w, h, &mut buf), m);
                    if c < best.1 {
                        best = (m, c);
                    }
                }
            }
        }
        best.0
    }

    fn inter_mode_bits(&self, fd: &FrameDec, y_mode: u8) -> f64 {
        let ctx = fd.b.mode_context[fd.b.ref_frame[0] as usize] as usize;
        let p = fd.probs.inter_mode[ctx];
        tree_bits(&INTER_MODE_TREE, y_mode - NEARESTMV, |n| p[n])
    }

    /// The bits of the single reference `rf` (read_ref_frames).
    fn ref_bits(&self, fd: &FrameDec, rf: i8) -> f64 {
        let mut c = BitCounter::default();
        write_single_ref(&mut c, fd, rf);
        c.bits
    }

    /// The best inter mode and motion vector from the search reference
    /// `ri`, by SAD and estimated bits; and that cost.
    fn inter_candidate(&self, fd: &mut FrameDec, ri: usize) -> (u8, Mv, f64) {
        fd.b.ref_frame = [self.refs[ri].ref_frame, NONE];
        fd.b.is_inter = true;
        fd.find_best_ref_mvs(0);
        let searched = self.motion_search(fd, ri);
        let rbits = self.ref_bits(fd, fd.b.ref_frame[0]);
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
                // The decoder would clamp it; still valid, but the SAD here
                // would not be the prediction's.
                continue;
            }
            let mut bits = self.inter_mode_bits(fd, mode) + rbits;
            if mode == NEWMV {
                bits += mv_bits([mv[0] - fd.b.best_mv[0][0], mv[1] - fd.b.best_mv[0][1]]);
            }
            let c = self.block_sad_sub(fd, ri, mv, w, h, &mut buf) as f64 + lam * bits;
            if c < best.2 {
                best = (mode, mv, c);
            }
        }
        best
    }

    // -----------------------------------------------------------------
    // The block: decide, reconstruct, write.

    /// The largest transform size of a block of `bsize` under the frame's
    /// transform mode: what a block that does not code its size has.
    fn max_tx(&self, bsize: u8) -> u8 {
        MAX_TXSIZE_LOOKUP[bsize as usize].min(TX_MODE_TO_BIGGEST_TX_SIZE[self.h.tx_mode as usize])
    }

    fn setup_block(&self, fd: &mut FrameDec, r: u32, c: u32, bsize: u8) {
        let avail_u = r > 0;
        let avail_l = c > fd.mi_col_start;
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
            tx_size: self.max_tx(bsize),
            ref_frame: [INTRA_FRAME, NONE],
            ..Block::default()
        };
    }

    fn block<S: Sink>(&mut self, e: &mut S, fd: &mut FrameDec, r: u32, c: u32, bsize: u8) {
        self.setup_block(fd, r, c, bsize);
        let choice = match self.next_replay() {
            Some(Decision::Block(choice)) => choice,
            None => self.decide(fd, r, c, bsize),
            Some(d) => unreachable!("replay out of step: {d:?}"),
        };
        self.decisions.push(Decision::Block(choice));
        self.code_block(e, fd, choice);
    }

    /// Chooses the block's coding: the intra and inter candidates (per
    /// transform size tried) coded for real, the cheapest kept.
    fn decide(&mut self, fd: &mut FrameDec, r: u32, c: u32, bsize: u8) -> Choice {
        let max_tx = self.max_tx(bsize);
        let tx_set: Vec<u8> = if self.h.tx_mode == TX_MODE_SELECT {
            (max_tx.saturating_sub(self.effort.tx_depth)..=max_tx)
                .rev()
                .collect()
        } else {
            vec![max_tx]
        };
        let mut cands = self.intra_candidates(fd, &tx_set);
        if !self.h.frame_is_intra {
            // The best reference by SAD; its candidates coded for real.
            let mut best: Option<(i8, u8, Mv, f64)> = None;
            for ri in 0..self.refs.len() {
                let (y_mode, mv, cost) = self.inter_candidate(fd, ri);
                if best.is_none_or(|b| cost < b.3) {
                    best = Some((self.refs[ri].ref_frame, y_mode, mv, cost));
                }
            }
            if let Some((ref_frame, y_mode, mv, _)) = best {
                cands.extend(tx_set.iter().map(|&t| Choice {
                    is_inter: true,
                    y_mode,
                    uv_mode: DC_PRED,
                    mv,
                    ref_frame,
                    tx_size: t,
                }));
            }
        }
        if cands.len() == 1 {
            return cands[0];
        }
        let bw8 = NUM_8X8_WIDE[bsize as usize] as u32;
        let bh8 = NUM_8X8_HIGH[bsize as usize] as u32;
        let pre = self.snapshot(fd, r, c, bw8, bh8);
        let record = std::mem::replace(&mut self.record, false);
        let mut best = (cands[0], f64::MAX);
        for &cand in &cands {
            self.setup_block(fd, r, c, bsize);
            let mut ctr = BitCounter::default();
            self.code_block(&mut ctr, fd, cand);
            let cost = self.region_sse(fd, r, c, bw8, bh8) + self.lambda * ctr.bits;
            if cost < best.1 {
                best = (cand, cost);
            }
            self.restore(fd, &pre);
        }
        self.record = record;
        self.setup_block(fd, r, c, bsize);
        best.0
    }

    /// Codes the current block (set up by `setup_block`) with `choice`:
    /// reconstructs it, writes its mode info and tokens, and records what
    /// the decoder remembers of it.
    fn code_block<S: Sink>(&mut self, e: &mut S, fd: &mut FrameDec, choice: Choice) {
        let bsize = fd.b.mi_size;
        let (r, c) = (fd.b.mi_row, fd.b.mi_col);
        fd.b.tx_size = choice.tx_size;
        let codings: [PlaneCoding; 3] = if choice.is_inter {
            fd.b.is_inter = true;
            fd.b.ref_frame = [choice.ref_frame, NONE];
            fd.find_best_ref_mvs(0);
            fd.b.y_mode = choice.y_mode;
            fd.b.interp_filter = EIGHTTAP;
            fd.b.block_mvs = [[choice.mv; 4], [[0; 2]; 4]];
            let mut codings: [PlaneCoding; 3] = Default::default();
            for (plane, coding) in codings.iter_mut().enumerate() {
                let (x, y, w, h, _) = self.plane_region(fd, plane);
                fd.predict_inter(plane, x, y, w, h, 0)
                    .expect("reference present");
                *coding = self.code_plane(fd, plane, false).0;
            }
            codings
        } else {
            fd.b.is_inter = false;
            fd.b.ref_frame = [INTRA_FRAME, NONE];
            fd.b.block_mvs = [[[0; 2]; 4]; 2];
            fd.b.interp_filter = 0;
            fd.b.y_mode = choice.y_mode;
            fd.b.sub_modes = [choice.y_mode; 4];
            fd.b.uv_mode = choice.uv_mode;
            [
                self.code_plane(fd, 0, true).0,
                self.code_plane(fd, 1, true).0,
                self.code_plane(fd, 2, true).0,
            ]
        };
        let skip = codings
            .iter()
            .all(|p| p.iter().all(|t| t.is_none_or(|t| t.eob == 0)));
        fd.b.skip = skip;
        let avail_u = fd.b.avail_u;
        let avail_l = fd.b.avail_l;
        // Mode info.
        let skip_ctx =
            fd.b.above.map_or(0, |m| m.skip as usize) + fd.b.left.map_or(0, |m| m.skip as usize);
        e.write(skip, fd.probs.skip[skip_ctx]);
        if self.h.frame_is_intra {
            write_tx_size(e, fd, true);
            let am = fd.b.above.map_or(DC_PRED, |m| m.sub_modes[2]);
            let lm = fd.b.left.map_or(DC_PRED, |m| m.sub_modes[1]);
            let p = KF_Y_MODE_PROBS[am as usize][lm as usize];
            e.tree(&INTRA_MODE_TREE, fd.b.y_mode, |n| p[n]);
            let p = KF_UV_MODE_PROBS[fd.b.y_mode as usize];
            e.tree(&INTRA_MODE_TREE, fd.b.uv_mode, |n| p[n]);
        } else {
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
            // An inter block without residual does not code its transform
            // size: it has the largest.
            if skip && fd.b.is_inter {
                fd.b.tx_size = self.max_tx(bsize);
            } else {
                write_tx_size(e, fd, true);
            }
            if fd.b.is_inter {
                write_single_ref(e, fd, fd.b.ref_frame[0]);
                let mctx = fd.b.mode_context[fd.b.ref_frame[0] as usize] as usize;
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
            let (base_x, base_y, pw, ph, _) = self.plane_region(fd, plane);
            if skip {
                // No tokens: every transform block's context is 0.
                fd.above_nonzero[plane][base_x >> 2..(base_x + pw) >> 2].fill(0);
                fd.left_nonzero[plane][base_y >> 2..(base_y + ph) >> 2].fill(0);
                continue;
            }
            let tx_sz = if plane > 0 {
                fd.uv_tx_size()
            } else {
                fd.b.tx_size
            };
            let step = 1usize << tx_sz;
            let n4w = pw / 4;
            for (k, tb) in coding.iter().enumerate() {
                let x = (k % n4w.div_ceil(step)) * step;
                let y = (k / n4w.div_ceil(step)) * step;
                let start_x = base_x + 4 * x;
                let start_y = base_y + 4 * y;
                let mut nonzero = false;
                if let Some(tb) = tb {
                    let stats = if self.record {
                        Some(&mut *self.stats)
                    } else {
                        None
                    };
                    write_tokens(e, fd, stats, plane, start_x, start_y, tx_sz, &tb);
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
                let at = fd.mi_idx(y, x);
                fd.mi[at] = info;
            }
        }
    }
}

#[inline]
fn eob_of(scan: &[u16], coefs: &[i32]) -> usize {
    scan.iter()
        .rposition(|&p| coefs[p as usize] != 0)
        .map_or(0, |i| i + 1)
}

#[inline]
fn bits_of(scan: &[u16], coefs: &[i32]) -> f64 {
    let mut bits = 0.0;
    for &p in scan {
        bits += coef_bits(coefs[p as usize]);
    }
    bits
}

/// The estimated bits of a coefficient in the scan before the end of
/// block: 1.5 for a zero, else `3 + 2 log2(1 + |v|)` (a table for the
/// common magnitudes, the same values).
#[inline]
fn coef_bits(v: i32) -> f64 {
    static TABLE: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        (0..1024u32)
            .map(|m| {
                if m == 0 {
                    1.5
                } else {
                    3.0 + 2.0 * (1.0 + m as f64).log2()
                }
            })
            .collect()
    });
    let m = v.unsigned_abs();
    match t.get(m as usize) {
        Some(&b) => b,
        None => 3.0 + 2.0 * (1.0 + m as f64).log2(),
    }
}

/// The mode info columns of tile column `tile_col` (get_tile_offset).
pub(crate) fn column_bounds(h: &FrameHeader, tile_col: u32) -> (u32, u32) {
    let off = |n: u32| {
        let sbs = h.mi_cols.div_ceil(8);
        (((n * sbs) >> h.tile_cols_log2) << 3).min(h.mi_cols)
    };
    (off(tile_col), off(tile_col + 1))
}

/// The quantiser tables' index for the frame's bit depth: 0, 1, 2 for 8,
/// 10, 12 bits.
fn bd_index(h: &FrameHeader) -> usize {
    ((h.bit_depth - 8) >> 1) as usize
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
        DC_QLOOKUP[bd_index(h)][(q + dcd).clamp(0, 255) as usize],
        AC_QLOOKUP[bd_index(h)][(q + acd).clamp(0, 255) as usize],
    )
}

fn scan_for(tx_sz: u8, tx_type: u8) -> &'static [u16] {
    crate::decoder::block::scan(tx_sz, tx_type)
}

/// read_ref_frames()'s inverse for a single reference `rf` (the frame's
/// reference mode is SINGLE_REFERENCE).
fn write_single_ref(e: &mut impl Sink, fd: &FrameDec, rf: i8) {
    let ctx = fd.single_ref_p1_ctx();
    e.write(rf != LAST_FRAME, fd.probs.single_ref[ctx][0]);
    if rf != LAST_FRAME {
        let ctx = fd.single_ref_p2_ctx();
        e.write(rf == ALTREF_FRAME, fd.probs.single_ref[ctx][1]);
    }
}

/// The inverse of read_tx_size (6.4.10) for the current block.
fn write_tx_size(e: &mut impl Sink, fd: &FrameDec, allow_select: bool) {
    let max_tx = MAX_TXSIZE_LOOKUP[fd.b.mi_size as usize];
    if !(allow_select && fd.h.tx_mode == TX_MODE_SELECT && fd.b.mi_size >= BLOCK_8X8) {
        return;
    }
    let mut above = max_tx;
    let mut left = max_tx;
    if let Some(m) = fd.b.above
        && !m.skip
    {
        above = m.tx_size;
    }
    if let Some(m) = fd.b.left
        && !m.skip
    {
        left = m.tx_size;
    }
    if !fd.b.avail_l {
        left = above;
    }
    if !fd.b.avail_u {
        above = left;
    }
    let ctx = ((above + left) > max_tx) as usize;
    let p = fd.probs.tx[max_tx as usize][ctx];
    let tree: &[i8] = match max_tx {
        TX_32X32 => &TX_SIZE_32_TREE,
        TX_16X16 => &TX_SIZE_16_TREE,
        _ => &TX_SIZE_8_TREE,
    };
    e.tree(tree, fd.b.tx_size, |n| p[n]);
}

/// The inverse of tokens() (6.4.24): the same contexts, writing.
#[allow(clippy::too_many_arguments)]
fn write_tokens(
    e: &mut impl Sink,
    fd: &mut FrameDec,
    mut stats: Option<&mut CoefStats>,
    plane: usize,
    start_x: usize,
    start_y: usize,
    tx_sz: u8,
    tb: &TxBlock<'_>,
) {
    let scan = scan_for(tx_sz, tb.tx_type);
    let seg_eob = 16usize << (tx_sz << 1);
    let ref_type = fd.b.is_inter as usize;
    let ptype = (plane > 0) as usize;
    let txs = tx_sz as usize;
    let (sx, sy) = if plane > 0 {
        (fd.ss_x, fd.ss_y)
    } else {
        (0, 0)
    };
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
        let mut st = stats
            .as_deref_mut()
            .map(|s| &mut s[txs][ptype][ref_type][band][ctx]);
        if check_eob {
            let more = c < tb.eob;
            if let Some(st) = st.as_deref_mut() {
                st[0][more as usize] += 1;
            }
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
        if let Some(st) = st {
            st[1][(token != ZERO_TOKEN) as usize] += 1;
            if token != ZERO_TOKEN {
                st[2][(token > 1) as usize] += 1;
            }
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
                if token == DCT_VAL_CAT6 {
                    // The high bits above 8-bit range, most significant
                    // first, each with probability 255 (read_coef).
                    let bd = fd.bit_depth as i32;
                    for k in 0..bd - 8 {
                        e.write((extra >> (num_extra + bd - 9 - k)) & 1 != 0, 255);
                    }
                }
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
fn write_mv(e: &mut impl Sink, fd: &FrameDec, d: Mv) {
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
