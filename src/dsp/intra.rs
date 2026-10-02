//! The intra predictors of section 8.5.1, given the edge arrays.

// Loops index arrays the way the specification's formulas do.
#![allow(clippy::needless_range_loop)]

use crate::consts::*;

#[inline]
fn r2(x: i32, n: u32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// Fills `dst` (row stride `stride`) with the `1 << log2` square
/// prediction for `mode`.
///
/// `above` holds aboveRow[-1..2*size-1]: `above[0]` is aboveRow[-1] and
/// `above[1 + i]` is aboveRow[i]. `left` holds leftCol[0..size-1].
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict(
    mode: u8,
    log2: u32,
    above: &[i32],
    left: &[i32],
    have_left: bool,
    have_above: bool,
    bit_depth: u32,
    dst: &mut [u16],
    stride: usize,
) {
    let size = 1usize << log2;
    let a = |i: isize| above[(i + 1) as usize];
    let l = |i: usize| left[i];
    let mut pred = [[0i32; 32]; 32];
    match mode {
        V_PRED => {
            for row in pred.iter_mut().take(size) {
                for j in 0..size {
                    row[j] = a(j as isize);
                }
            }
        }
        H_PRED => {
            for (i, row) in pred.iter_mut().enumerate().take(size) {
                for v in row.iter_mut().take(size) {
                    *v = l(i);
                }
            }
        }
        D207_PRED => {
            for j in 0..size {
                pred[size - 1][j] = l(size - 1);
            }
            for i in 0..size - 1 {
                pred[i][0] = r2(l(i) + l(i + 1), 1);
            }
            for i in 0..size.saturating_sub(2) {
                pred[i][1] = r2(l(i) + 2 * l(i + 1) + l(i + 2), 2);
            }
            pred[size - 2][1] = r2(l(size - 2) + 3 * l(size - 1), 2);
            for j in 2..size {
                for i in (0..size - 1).rev() {
                    pred[i][j] = pred[i + 1][j - 2];
                }
            }
        }
        D45_PRED => {
            for i in 0..size {
                for j in 0..size {
                    let k = (i + j) as isize;
                    pred[i][j] = if i + j + 2 < size * 2 {
                        r2(a(k) + a(k + 1) * 2 + a(k + 2), 2)
                    } else {
                        a(2 * size as isize - 1)
                    };
                }
            }
        }
        D63_PRED => {
            for i in 0..size {
                for j in 0..size {
                    let k = (i / 2 + j) as isize;
                    pred[i][j] = if i & 1 != 0 {
                        r2(a(k) + a(k + 1) * 2 + a(k + 2), 2)
                    } else {
                        r2(a(k) + a(k + 1), 1)
                    };
                }
            }
        }
        D117_PRED => {
            for j in 0..size {
                pred[0][j] = r2(a(j as isize - 1) + a(j as isize), 1);
            }
            pred[1][0] = r2(l(0) + 2 * a(-1) + a(0), 2);
            for j in 1..size {
                pred[1][j] = r2(a(j as isize - 2) + 2 * a(j as isize - 1) + a(j as isize), 2);
            }
            pred[2][0] = r2(a(-1) + 2 * l(0) + l(1), 2);
            for i in 3..size {
                pred[i][0] = r2(l(i - 3) + 2 * l(i - 2) + l(i - 1), 2);
            }
            for i in 2..size {
                for j in 1..size {
                    pred[i][j] = pred[i - 2][j - 1];
                }
            }
        }
        D135_PRED => {
            pred[0][0] = r2(l(0) + 2 * a(-1) + a(0), 2);
            for j in 1..size {
                pred[0][j] = r2(a(j as isize - 2) + 2 * a(j as isize - 1) + a(j as isize), 2);
            }
            pred[1][0] = r2(a(-1) + 2 * l(0) + l(1), 2);
            for i in 2..size {
                pred[i][0] = r2(l(i - 2) + 2 * l(i - 1) + l(i), 2);
            }
            for i in 1..size {
                for j in 1..size {
                    pred[i][j] = pred[i - 1][j - 1];
                }
            }
        }
        D153_PRED => {
            pred[0][0] = r2(l(0) + a(-1), 1);
            for i in 1..size {
                pred[i][0] = r2(l(i - 1) + l(i), 1);
            }
            pred[0][1] = r2(l(0) + 2 * a(-1) + a(0), 2);
            pred[1][1] = r2(a(-1) + 2 * l(0) + l(1), 2);
            for i in 2..size {
                pred[i][1] = r2(l(i - 2) + 2 * l(i - 1) + l(i), 2);
            }
            for j in 2..size {
                pred[0][j] = r2(
                    a(j as isize - 3) + 2 * a(j as isize - 2) + a(j as isize - 1),
                    2,
                );
            }
            for i in 1..size {
                for j in 2..size {
                    pred[i][j] = pred[i - 1][j - 2];
                }
            }
        }
        TM_PRED => {
            let max = (1i32 << bit_depth) - 1;
            for i in 0..size {
                for j in 0..size {
                    pred[i][j] = (a(j as isize) + l(i) - a(-1)).clamp(0, max);
                }
            }
        }
        _ => {
            // DC_PRED
            let v = if have_left && have_above {
                let mut sum = 0;
                for k in 0..size {
                    sum += l(k) + a(k as isize);
                }
                (sum + size as i32) >> (log2 + 1)
            } else if have_left {
                let sum: i32 = (0..size).map(l).sum();
                (sum + (1 << (log2 - 1))) >> log2
            } else if have_above {
                let sum: i32 = (0..size).map(|k| a(k as isize)).sum();
                (sum + (1 << (log2 - 1))) >> log2
            } else {
                1 << (bit_depth - 1)
            };
            for row in pred.iter_mut().take(size) {
                for x in row.iter_mut().take(size) {
                    *x = v;
                }
            }
        }
    }
    for i in 0..size {
        let d = &mut dst[i * stride..i * stride + size];
        for j in 0..size {
            d[j] = pred[i][j] as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(mode: u8, above: &[i32], left: &[i32]) -> Vec<u16> {
        let mut dst = vec![0u16; 16];
        predict(mode, 2, above, left, true, true, 8, &mut dst, 4);
        dst
    }

    #[test]
    fn simple_modes() {
        let above = [100, 10, 20, 30, 40, 50, 60, 70, 80];
        let left = [1, 2, 3, 4];
        assert_eq!(&run(V_PRED, &above, &left)[4..8], &[10, 20, 30, 40]);
        assert_eq!(&run(H_PRED, &above, &left)[4..8], &[2, 2, 2, 2]);
        // DC: (10+20+30+40 + 1+2+3+4 + 4) >> 3 = 114 >> 3 = 14.
        assert!(run(DC_PRED, &above, &left).iter().all(|&v| v == 14));
        // TM: clip(above + left - topleft): 10 + 1 - 100 < 0.
        assert_eq!(run(TM_PRED, &above, &left)[0], 0);
    }

    #[test]
    fn directional_modes_are_constant_on_constant_edges() {
        let above = [77; 9];
        let left = [77; 4];
        for mode in 0..10 {
            assert!(
                run(mode, &above, &left).iter().all(|&v| v == 77),
                "mode {mode}"
            );
        }
    }
}
