//! Block inter prediction (8.5.2.4): the separable 8-tap subpixel filter
//! over a reference plane, with the reference's edge samples repeated
//! outward (the clamps of the specification).

use crate::tables::SUBPEL_FILTERS;

/// A reference plane as the predictor sees it.
pub(crate) struct RefPlane<'a> {
    pub data: &'a [u16],
    pub stride: usize,
    /// lastX / lastY: the bottom-right sample of the plane.
    pub last_x: i32,
    pub last_y: i32,
}

/// Predicts a `w` x `h` block whose top-left is at (`x`, `y`) in 1/16
/// sample units of the reference, stepping `x_step` / `y_step` sixteenths
/// per output sample. Writes `w * h` samples to `out` (stride `w`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict(
    r: &RefPlane,
    x: i32,
    y: i32,
    x_step: i32,
    y_step: i32,
    w: usize,
    h: usize,
    filter: u8,
    bit_depth: u32,
    out: &mut [u16],
) {
    let max = (1i32 << bit_depth) - 1;
    let taps = &SUBPEL_FILTERS[filter as usize];
    let ih = ((((h as i32 - 1) * y_step + 15) >> 4) + 8) as usize;
    let mut stack = [0i32; 64 * 72];
    let mut heap;
    let inter: &mut [i32] = if ih * w <= stack.len() {
        &mut stack
    } else {
        heap = vec![0i32; ih * w];
        &mut heap
    };
    let y0 = (y >> 4) - 3;
    let x0 = (x >> 4) - 3;
    // Unscaled, and the whole footprint inside the plane: no clamps.
    let inside = x_step == 16
        && y_step == 16
        && x0 >= 0
        && y0 >= 0
        && x0 + w as i32 + 7 <= r.last_x
        && y0 + ih as i32 <= r.last_y;
    if inside {
        let fx = &taps[(x & 15) as usize];
        for row in 0..ih {
            let base = (y0 as usize + row) * r.stride + x0 as usize;
            let src = &r.data[base..base + w + 7];
            let dst = &mut inter[row * w..row * w + w];
            if x & 15 == 0 {
                for c in 0..w {
                    dst[c] = src[c + 3] as i32;
                }
            } else {
                for c in 0..w {
                    let s = &src[c..c + 8];
                    let mut sum = 0;
                    for t in 0..8 {
                        sum += fx[t] * s[t] as i32;
                    }
                    dst[c] = ((sum + 64) >> 7).clamp(0, max);
                }
            }
        }
        let fy = &taps[(y & 15) as usize];
        for rr in 0..h {
            let o = &mut out[rr * w..rr * w + w];
            if y & 15 == 0 {
                for c in 0..w {
                    o[c] = inter[(rr + 3) * w + c] as u16;
                }
            } else {
                for c in 0..w {
                    let mut sum = 0;
                    for t in 0..8 {
                        sum += fy[t] * inter[(rr + t) * w + c];
                    }
                    o[c] = ((sum + 64) >> 7).clamp(0, max) as u16;
                }
            }
        }
        return;
    }
    // The general case, exactly as written.
    for row in 0..ih {
        let ry = (y0 + row as i32).clamp(0, r.last_y) as usize;
        let src = &r.data[ry * r.stride..];
        for c in 0..w {
            let p = x + x_step * c as i32;
            let f = &taps[(p & 15) as usize];
            let px = (p >> 4) - 3;
            let mut sum = 0;
            for t in 0..8 {
                let sx = (px + t as i32).clamp(0, r.last_x) as usize;
                sum += f[t] * src[sx] as i32;
            }
            inter[row * w + c] = ((sum + 64) >> 7).clamp(0, max);
        }
    }
    for rr in 0..h {
        let p = (y & 15) + y_step * rr as i32;
        let f = &taps[(p & 15) as usize];
        let base = (p >> 4) as usize;
        for c in 0..w {
            let mut sum = 0;
            for t in 0..8 {
                sum += f[t] * inter[(base + t) * w + c];
            }
            out[rr * w + c] = ((sum + 64) >> 7).clamp(0, max) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane() -> Vec<u16> {
        (0..32 * 32).map(|i| ((i * 37) % 251) as u16).collect()
    }

    #[test]
    fn integer_positions_copy() {
        let d = plane();
        let r = RefPlane { data: &d, stride: 32, last_x: 31, last_y: 31 };
        let mut out = vec![0u16; 64];
        predict(&r, 5 * 16, 7 * 16, 16, 16, 8, 8, 0, 8, &mut out);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out[y * 8 + x], d[(7 + y) * 32 + 5 + x]);
            }
        }
    }

    #[test]
    fn fast_path_matches_general_path() {
        let d = plane();
        let r = RefPlane { data: &d, stride: 32, last_x: 31, last_y: 31 };
        for filter in 0..4 {
            for frac in 0..16 {
                let (x, y) = (8 * 16 + frac, 9 * 16 + (15 - frac));
                let mut a = vec![0u16; 64];
                predict(&r, x, y, 16, 16, 8, 8, filter, 8, &mut a);
                // Force the general path by claiming a tiny plane bound
                // that the footprint never reaches anyway... instead compare
                // against a direct evaluation of the formula.
                let mut b = vec![0u16; 64];
                let ih = ((7 * 16 + 15) >> 4) + 8;
                let mut inter = vec![0i32; ih * 8];
                for rr in 0..ih {
                    for c in 0..8 {
                        let p = x + 16 * c as i32;
                        let mut s = 0;
                        for t in 0..8 {
                            let sy = ((y >> 4) + rr as i32 - 3).clamp(0, 31) as usize;
                            let sx = ((p >> 4) + t as i32 - 3).clamp(0, 31) as usize;
                            s += SUBPEL_FILTERS[filter as usize][(p & 15) as usize][t] * d[sy * 32 + sx] as i32;
                        }
                        inter[rr * 8 + c] = ((s + 64) >> 7).clamp(0, 255);
                    }
                }
                for rr in 0..8 {
                    for c in 0..8 {
                        let p = (y & 15) + 16 * rr as i32;
                        let mut s = 0;
                        for t in 0..8 {
                            s += SUBPEL_FILTERS[filter as usize][(p & 15) as usize][t] * inter[((p >> 4) as usize + t) * 8 + c];
                        }
                        b[rr * 8 + c] = ((s + 64) >> 7).clamp(0, 255) as u16;
                    }
                }
                assert_eq!(a, b, "filter {filter} frac {frac}");
            }
        }
    }

    #[test]
    fn filters_sum_to_128() {
        for f in SUBPEL_FILTERS.iter() {
            for phase in f.iter() {
                assert_eq!(phase.iter().sum::<i32>(), 128);
            }
        }
    }
}
