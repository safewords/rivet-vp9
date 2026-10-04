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

/// Applies the loop filter to the frame, the planes in parallel when
/// `threads` allows (they are filtered independently).
pub(crate) fn filter_frame(
    h: &FrameHeader,
    lf: &LoopFilter,
    seg: &Segmentation,
    mi: &[MiInfo],
    planes: &mut [PlaneBuf; 3],
    threads: usize,
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
    let level = crate::dsp::level();
    let ctx = Ctx {
        h,
        mi,
        lvl: &lvl,
        params: &params,
        level,
    };
    let bands = h.sb64_rows as usize;
    if threads > 1 && bands > 1 {
        filter_bands(&ctx, planes, threads);
        return;
    }
    let mut edges = lf::Edges::new();
    for (plane, buf) in planes.iter_mut().enumerate() {
        let stride = buf.stride;
        let mut row = 0;
        while row < h.mi_rows {
            let mut col = 0;
            while col < h.mi_cols {
                for pass in 0..2 {
                    let g = superblock(&ctx, plane, pass, row, col, &mut edges);
                    g.filter(&ctx, &mut buf.data, stride, 0, 0, &edges);
                }
                col += 8;
            }
            row += 8;
        }
    }
}

/// Where a superblock pass's edges are ([`superblock`]).
struct Geometry {
    /// Plane coordinates of the superblock.
    x0: usize,
    y0: usize,
    vertical: bool,
    n_edges: usize,
    n_runs: usize,
}

impl Geometry {
    /// Filters edges `first..` of `e` in `buf` (row stride `stride`) whose
    /// row 0 is plane row `y_origin`.
    fn filter(
        &self,
        ctx: &Ctx,
        buf: &mut [u16],
        stride: usize,
        y_origin: usize,
        first: usize,
        e: &lf::Edges,
    ) {
        lf::filter_edges(
            ctx.level,
            buf,
            stride,
            self.x0,
            self.y0 - y_origin,
            self.vertical,
            first,
            self.n_edges,
            self.n_runs,
            e,
            ctx.h.bit_depth,
        );
    }
}

/// What every superblock's filtering reads.
struct Ctx<'a> {
    h: &'a FrameHeader,
    mi: &'a [MiInfo],
    lvl: &'a [[[u8; 2]; 4]; MAX_SEGMENTS],
    params: &'a [(i32, i32, i32); 64],
    level: crate::dsp::Level,
}

/// Rows above a superblock row that its filtering reads and writes: the
/// horizontal edges at its top reach 8 samples up.
const HALO: usize = 8;

/// The bottom rows of a band (superblock row) as final as the band's own
/// filtering leaves them, for the band below: the columns before `done`.
struct Halo {
    done: [usize; 3],
    rows: [Vec<u16>; 3],
}

/// The loop filter with the superblock rows ("bands") on several threads,
/// as a wavefront.
///
/// Superblock (r, c)'s filtering reads and writes the samples from 8 above
/// and 8 left of it to its bottom-right corner; (r, c + 1)'s starts 8 left
/// of column c + 1. So once (r, c + 1) is done, nothing of row r touches
/// the columns before (c + 2) * 64 - 8 again, and (r + 1, c) can run: the
/// order of every overlapping pair of superblocks is the specification's
/// raster order, and the result is the serial filter's.
///
/// Each band filters its own rows in place and the 8 rows above it (the
/// "halo") in a copy, which it receives from the band above column range
/// by column range as the band above finishes with them. Only the
/// horizontal edges at the top of a band reach into the halo; they are
/// filtered in a small buffer of the halo and the band's first rows. The
/// halos are written back in band order once every band is done.
fn filter_bands(ctx: &Ctx, planes: &mut [PlaneBuf; 3], threads: usize) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex};
    let h = ctx.h;
    let bands = h.sb64_rows as usize;
    let sb_cols = h.sb64_cols as usize;
    let sub = |p: usize| -> (u32, u32) {
        if p > 0 {
            (h.subsampling_x, h.subsampling_y)
        } else {
            (0, 0)
        }
    };
    let strides = [planes[0].stride, planes[1].stride, planes[2].stride];
    let band_h = |p: usize| 64usize >> sub(p).1;
    // Boundary k: the bottom rows of band k, for band k + 1.
    let boundaries: Vec<(Mutex<Halo>, Condvar)> = (0..bands.saturating_sub(1))
        .map(|_| {
            (
                Mutex::new(Halo {
                    done: [0; 3],
                    rows: strides.map(|s| vec![0u16; HALO * s]),
                }),
                Condvar::new(),
            )
        })
        .collect();
    // Each band's rows of each plane, to hand to the band's task.
    let [py, pu, pv] = planes;
    let chunks: Vec<Mutex<Option<[&mut [u16]; 3]>>> = py
        .data
        .chunks_mut(band_h(0) * strides[0])
        .zip(pu.data.chunks_mut(band_h(1) * strides[1]))
        .zip(pv.data.chunks_mut(band_h(2) * strides[2]))
        .map(|((y, u), v)| Mutex::new(Some([y, u, v])))
        .collect();
    let next = AtomicUsize::new(0);
    let band = |k: usize| -> [Vec<u16>; 3] {
        let rows = chunks[k]
            .lock()
            .expect("unpoisoned")
            .take()
            .expect("each band once");
        let mut halo: [Vec<u16>; 3] = strides.map(|s| vec![0u16; HALO * s]);
        let mut copied = [0usize; 3];
        let mut published = [0usize; 3];
        let mut edges = lf::Edges::new();
        let mut top = vec![0u16; 2 * HALO * 64];
        for c in 0..sb_cols {
            if k > 0 {
                let (lock, cv) = &boundaries[k - 1];
                let mut b = lock.lock().expect("unpoisoned");
                for p in 0..3 {
                    let need = ((64 * (c + 1)) >> sub(p).0).min(strides[p]);
                    while b.done[p] < need {
                        b = cv.wait(b).expect("unpoisoned");
                    }
                    let (from, to, s) = (copied[p], b.done[p], strides[p]);
                    for r in 0..HALO {
                        halo[p][r * s + from..r * s + to]
                            .copy_from_slice(&b.rows[p][r * s + from..r * s + to]);
                    }
                    copied[p] = to;
                }
            }
            for p in 0..3 {
                let s = strides[p];
                let origin = k * band_h(p);
                for pass in 0..2 {
                    let g = superblock(ctx, p, pass, (k * 8) as u32, (c * 8) as u32, &mut edges);
                    let mut first = 0;
                    if pass == 1 && k > 0 {
                        // Edge 0 is the band's top: through the halo.
                        first = 1;
                        if edges.fs[0][..g.n_runs].iter().any(|&f| f != lf::SKIP) {
                            let w = 4 * g.n_runs;
                            let t = &mut top[..2 * HALO * w];
                            for r in 0..HALO {
                                t[r * w..r * w + w]
                                    .copy_from_slice(&halo[p][r * s + g.x0..r * s + g.x0 + w]);
                                t[(HALO + r) * w..(HALO + r) * w + w]
                                    .copy_from_slice(&rows[p][r * s + g.x0..r * s + g.x0 + w]);
                            }
                            let tg = Geometry {
                                x0: 0,
                                y0: HALO,
                                n_edges: 1,
                                ..g
                            };
                            tg.filter(ctx, t, w, 0, 0, &edges);
                            for r in 0..HALO {
                                halo[p][r * s + g.x0..r * s + g.x0 + w]
                                    .copy_from_slice(&t[r * w..r * w + w]);
                                rows[p][r * s + g.x0..r * s + g.x0 + w]
                                    .copy_from_slice(&t[(HALO + r) * w..(HALO + r) * w + w]);
                            }
                        }
                    }
                    g.filter(ctx, rows[p], s, origin, first, &edges);
                }
            }
            if k + 1 < bands {
                let (lock, cv) = &boundaries[k];
                let mut b = lock.lock().expect("unpoisoned");
                for p in 0..3 {
                    let s = strides[p];
                    let done = if c + 1 == sb_cols {
                        s
                    } else {
                        ((64 * (c + 1)) >> sub(p).0) - HALO
                    };
                    let from = published[p];
                    let base = band_h(p) - HALO;
                    for r in 0..HALO {
                        b.rows[p][r * s + from..r * s + done].copy_from_slice(
                            &rows[p][(base + r) * s + from..(base + r) * s + done],
                        );
                    }
                    published[p] = done;
                    b.done[p] = done;
                }
                cv.notify_all();
            }
        }
        halo
    };
    let workers = threads.min(bands);
    let mut halos: Vec<(usize, [Vec<u16>; 3])> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        if k >= bands {
                            break out;
                        }
                        out.push((k, band(k)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|t| t.join().expect("loop filter worker panicked"))
            .collect()
    });
    drop(chunks);
    halos.sort_by_key(|(k, _)| *k);
    for (k, halo) in halos.into_iter().skip(1) {
        for (p, rows) in halo.iter().enumerate() {
            let (s, start) = (strides[p], k * band_h(p) * strides[p]);
            planes[p].data[start - HALO * s..start].copy_from_slice(rows);
        }
    }
}

/// The edges of one pass of one plane of the superblock at mode info
/// (`row`, `col`): fills `e`, returns where they are.
fn superblock(
    ctx: &Ctx,
    plane: usize,
    pass: usize,
    row: u32,
    col: u32,
    e: &mut lf::Edges,
) -> Geometry {
    let (h, mi, lvl, params) = (ctx.h, ctx.mi, ctx.lvl, ctx.params);
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
    // Every decision below depends on the 8x8 block a sample's luma
    // position falls in (the even one, subsampled). A cell of 8 x 8 plane
    // samples has two edges 4 apart and two runs of 4 along each, all in
    // the same block: decide once per cell what depends only on the block,
    // then per edge and run what depends on their own position (onScreen,
    // the block and transform edges, the 16-wide filter at the last column
    // or row, the transform edge exception at an odd right edge).
    let n_edges = 16usize >> sub;
    let n_runs = edge_len as usize / 4;
    for row in e.fs.iter_mut().take(n_edges) {
        row[..n_runs].fill(lf::SKIP);
    }
    // Luma coordinates of edge `ed`, run `rn`.
    let at = |ed: u32, rn: u32| {
        if pass == 0 {
            (col * 8 + ed * (4 << sub_x), row * 8 + ((4 * rn) << sub_y))
        } else {
            (col * 8 + ((4 * rn) << sub_x), row * 8 + ed * (4 << sub_y))
        }
    };
    let (ux, uy) = if h.legacy_uv {
        (1, 1)
    } else {
        (h.subsampling_x as usize, h.subsampling_y as usize)
    };
    for ca in 0..(n_edges / 2) as u32 {
        for cb in 0..(n_runs / 2) as u32 {
            let (x, y) = at(2 * ca, 2 * cb);
            // onScreen (step 13): every other position of the cell is
            // further right or down.
            if x >= 8 * mi_cols || y >= 8 * mi_rows {
                continue;
            }
            let loop_col = ((x >> 3) >> sub_x) << sub_x;
            let loop_row = ((y >> 3) >> sub_y) << sub_y;
            let m = &mi[(loop_row * mi_cols + loop_col) as usize];
            // Adaptive filter strength (8.8.4).
            let mode = m.y_mode;
            let mode_type = (mode == NEARESTMV || mode == NEARMV || mode == NEWMV) as usize;
            let rf = m.ref_frame[0].max(0) as usize;
            let l = lvl[m.segment_id as usize][rf][mode_type];
            if l == 0 {
                continue;
            }
            let (limit, blimit, thresh) = params[l as usize];
            let mi_size = m.mi_size;
            let tx_sz = if plane > 0 {
                if mi_size < BLOCK_8X8 {
                    TX_4X4
                } else {
                    // As get_uv_tx_size (4:2:0's for a legacy stream).
                    let uv = SS_SIZE_LOOKUP[mi_size as usize][ux][uy];
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
            // Block sizes are powers of two: masks, not divisions.
            let block_mask = if pass == 0 {
                8 * NUM_8X8_WIDE[sb_size as usize] as u32 - 1
            } else {
                8 * NUM_8X8_HIGH[sb_size as usize] as u32 - 1
            };
            let coded = m.ref_frame[0] <= INTRA_FRAME || !m.skip;
            for eo in 0..2 {
                let ed = 2 * ca + eo;
                let (x, y) = at(ed, 2 * cb);
                if (pass == 0 && x == 0) || (pass == 1 && y == 0) {
                    continue;
                }
                let across = if pass == 0 { x } else { y };
                let is_block_edge = across & block_mask == 0;
                let tx_edge = ed & ((1 << tx_sz) - 1) == 0;
                if !is_block_edge && !(tx_edge && coded) {
                    continue;
                }
                // Filter size (8.8.3).
                let is_32_edge = ed % 8 == 0;
                let base_size = if tx_sz == TX_4X4 && is_32_edge {
                    TX_8X8
                } else {
                    tx_sz.min(TX_16X16)
                };
                let filter_size = if (pass == 0
                    && sub_x == 1
                    && base_size == TX_16X16
                    && (x >> 3) == mi_cols - 1)
                    || (pass == 1 && sub_y == 1 && base_size == TX_16X16 && (y >> 3) == mi_rows - 1)
                {
                    TX_8X8
                } else {
                    base_size
                };
                for k in 0..2 {
                    let rn = 2 * cb + k;
                    let (x, y) = at(ed, rn);
                    if k == 1 && (x >= 8 * mi_cols || y >= 8 * mi_rows) {
                        break;
                    }
                    let is_tx_edge = if pass == 1
                        && sub_x == 1
                        && mi_cols & 1 == 1
                        && ed & 1 == 1
                        && (x + 8) >= mi_cols * 8
                    {
                        false
                    } else {
                        tx_edge
                    };
                    if !(is_block_edge || (is_tx_edge && coded)) {
                        continue;
                    }
                    let (ed, rn) = (ed as usize, rn as usize);
                    e.fs[ed][rn] = filter_size as i8;
                    e.limit[ed][rn] = limit as u8;
                    e.blimit[ed][rn] = blimit as u8;
                    e.thresh[ed][rn] = thresh as u8;
                }
            }
        }
    }
    Geometry {
        x0: ((col * 8) >> sub_x) as usize,
        y0: ((row * 8) >> sub_y) as usize,
        vertical: pass == 0,
        n_edges,
        n_runs,
    }
}
