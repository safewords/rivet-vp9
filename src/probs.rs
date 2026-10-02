//! Frame contexts — the probability tables a frame decodes with — and the
//! symbol counts and backward adaptation of section 8.4.

// Loops index arrays the way the specification's formulas do.
#![allow(clippy::needless_range_loop)]

use crate::consts::*;
use crate::tables::*;

/// One frame context: every adaptive probability table (section 10.5's
/// "x" for each "default_x").
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Probs {
    /// `[maxTxSize][ctx][node]`; row 0 unused.
    pub tx: [[[u8; 3]; 2]; 4],
    pub coef: CoefProbs,
    pub skip: [u8; 3],
    pub inter_mode: [[u8; 3]; 7],
    pub interp_filter: [[u8; 2]; 4],
    pub is_inter: [u8; 4],
    pub comp_mode: [u8; 5],
    pub single_ref: [[u8; 2]; 5],
    pub comp_ref: [u8; 5],
    pub y_mode: [[u8; 9]; 4],
    pub uv_mode: [[u8; 9]; 10],
    pub partition: [[u8; 3]; 16],
    pub mv_joint: [u8; 3],
    pub mv_sign: [u8; 2],
    pub mv_class: [[u8; 10]; 2],
    pub mv_class0_bit: [u8; 2],
    pub mv_bits: [[u8; 10]; 2],
    pub mv_class0_fr: [[[u8; 3]; 2]; 2],
    pub mv_fr: [[u8; 3]; 2],
    pub mv_class0_hp: [u8; 2],
    pub mv_hp: [u8; 2],
}

impl Default for Probs {
    fn default() -> Self {
        Probs {
            tx: DEFAULT_TX_PROBS,
            coef: DEFAULT_COEF_PROBS,
            skip: DEFAULT_SKIP_PROB,
            inter_mode: DEFAULT_INTER_MODE_PROBS,
            interp_filter: DEFAULT_INTERP_FILTER_PROBS,
            is_inter: DEFAULT_IS_INTER_PROB,
            comp_mode: DEFAULT_COMP_MODE_PROB,
            single_ref: DEFAULT_SINGLE_REF_PROB,
            comp_ref: DEFAULT_COMP_REF_PROB,
            y_mode: DEFAULT_Y_MODE_PROBS,
            uv_mode: DEFAULT_UV_MODE_PROBS,
            partition: DEFAULT_PARTITION_PROBS,
            mv_joint: DEFAULT_MV_JOINT_PROBS,
            mv_sign: DEFAULT_MV_SIGN_PROB,
            mv_class: DEFAULT_MV_CLASS_PROBS,
            mv_class0_bit: DEFAULT_MV_CLASS0_BIT_PROB,
            mv_bits: DEFAULT_MV_BITS_PROB,
            mv_class0_fr: DEFAULT_MV_CLASS0_FR_PROBS,
            mv_fr: DEFAULT_MV_FR_PROBS,
            mv_class0_hp: DEFAULT_MV_CLASS0_HP_PROB,
            mv_hp: DEFAULT_MV_HP_PROB,
        }
    }
}

impl Probs {
    /// load_probs( ctx ): everything but tx_probs and skip_prob (7.1.2).
    pub(crate) fn load_except_tx_skip(&mut self, from: &Probs) {
        let tx = self.tx;
        let skip = self.skip;
        *self = from.clone();
        self.tx = tx;
        self.skip = skip;
    }

    /// load_probs2( ctx ): tx_probs and skip_prob only.
    pub(crate) fn load_tx_skip(&mut self, from: &Probs) {
        self.tx = from.tx;
        self.skip = from.skip;
    }
}

/// Counts per coefficient context, `[txSz][plane > 0][is_inter][band][ctx][value]`.
pub(crate) type CoefCounts<const N: usize> = [[[[[[u32; N]; 6]; 6]; 2]; 2]; 4];

/// The symbol counts of section 8.3.
#[derive(Clone, Default)]
pub(crate) struct Counts {
    pub intra_mode: [[u32; 10]; 4],
    pub uv_mode: [[u32; 10]; 10],
    pub partition: [[u32; 4]; 16],
    pub interp_filter: [[u32; 3]; 4],
    pub inter_mode: [[u32; 4]; 7],
    /// `[maxTxSize][ctx][tx_size]`.
    pub tx: [[[u32; 4]; 2]; 4],
    pub is_inter: [[u32; 2]; 4],
    pub comp_mode: [[u32; 2]; 5],
    pub single_ref: [[[u32; 2]; 2]; 5],
    pub comp_ref: [[u32; 2]; 5],
    pub skip: [[u32; 2]; 3],
    pub mv_joint: [u32; 4],
    pub mv_sign: [[u32; 2]; 2],
    pub mv_class: [[u32; 11]; 2],
    pub mv_class0_bit: [[u32; 2]; 2],
    pub mv_class0_fr: [[[u32; 4]; 2]; 2],
    pub mv_class0_hp: [[u32; 2]; 2],
    pub mv_bits: [[[u32; 2]; 10]; 2],
    pub mv_fr: [[u32; 4]; 2],
    pub mv_hp: [[u32; 2]; 2],
    pub token: CoefCounts<3>,
    pub more_coefs: CoefCounts<2>,
}

const COUNT_SAT: u32 = 20;
const MAX_UPDATE_FACTOR: u32 = 128;

/// merge_prob (8.4.1).
fn merge_prob(pre_prob: u8, ct0: u32, ct1: u32, count_sat: u32, max_update_factor: u32) -> u8 {
    let den = ct0 + ct1;
    let prob = if den == 0 {
        128
    } else {
        ((ct0 as u64 * 256 + (den as u64 >> 1)) / den as u64).clamp(1, 255) as u32
    };
    let count = den.min(count_sat);
    let factor = max_update_factor * count / count_sat;
    let v = pre_prob as u32 * (256 - factor) + prob * factor;
    ((v + 128) >> 8) as u8
}

/// merge_probs (8.4.2): adapts `probs` along `tree` from node `i`.
fn merge_probs(tree: &[i8], i: usize, probs: &mut [u8], counts: &[u32], count_sat: u32, upd: u32) -> u32 {
    let s = tree[i];
    let left = if s <= 0 { counts[(-s) as usize] } else { merge_probs(tree, s as usize, probs, counts, count_sat, upd) };
    let r = tree[i + 1];
    let right = if r <= 0 { counts[(-r) as usize] } else { merge_probs(tree, r as usize, probs, counts, count_sat, upd) };
    probs[i >> 1] = merge_prob(probs[i >> 1], left, right, count_sat, upd);
    left + right
}

fn adapt_probs(tree: &[i8], probs: &mut [u8], counts: &[u32]) {
    merge_probs(tree, 0, probs, counts, COUNT_SAT, MAX_UPDATE_FACTOR);
}

fn adapt_prob(prob: &mut u8, counts: &[u32; 2]) {
    *prob = merge_prob(*prob, counts[0], counts[1], COUNT_SAT, MAX_UPDATE_FACTOR);
}

/// adapt_coef_probs (8.4.3). `probs` holds the pre-frame probabilities
/// (load_probs has been done).
pub(crate) fn adapt_coef_probs(probs: &mut Probs, counts: &Counts, frame_is_intra: bool, last_frame_was_key: bool) {
    let update_factor = if frame_is_intra {
        112
    } else if last_frame_was_key {
        128
    } else {
        112
    };
    for t in 0..4 {
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..6 {
                    let max_l = if k == 0 { 3 } else { 6 };
                    for l in 0..max_l {
                        let p = &mut probs.coef[t][i][j][k][l];
                        merge_probs(&SMALL_TOKEN_TREE, 2, p, &counts.token[t][i][j][k][l], 24, update_factor);
                        merge_probs(&BINARY_TREE, 0, p, &counts.more_coefs[t][i][j][k][l], 24, update_factor);
                    }
                }
            }
        }
    }
}

/// adapt_noncoef_probs (8.4.4).
pub(crate) fn adapt_noncoef_probs(
    probs: &mut Probs,
    counts: &Counts,
    interp_switchable: bool,
    tx_mode_select: bool,
    allow_hp: bool,
) {
    for i in 0..4 {
        adapt_prob(&mut probs.is_inter[i], &counts.is_inter[i]);
    }
    for i in 0..5 {
        adapt_prob(&mut probs.comp_mode[i], &counts.comp_mode[i]);
    }
    for i in 0..5 {
        adapt_prob(&mut probs.comp_ref[i], &counts.comp_ref[i]);
    }
    for i in 0..5 {
        for j in 0..2 {
            adapt_prob(&mut probs.single_ref[i][j], &counts.single_ref[i][j]);
        }
    }
    for i in 0..7 {
        adapt_probs(&INTER_MODE_TREE, &mut probs.inter_mode[i], &counts.inter_mode[i]);
    }
    for i in 0..4 {
        adapt_probs(&INTRA_MODE_TREE, &mut probs.y_mode[i], &counts.intra_mode[i]);
    }
    for i in 0..10 {
        adapt_probs(&INTRA_MODE_TREE, &mut probs.uv_mode[i], &counts.uv_mode[i]);
    }
    for i in 0..16 {
        adapt_probs(&PARTITION_TREE, &mut probs.partition[i], &counts.partition[i]);
    }
    for i in 0..3 {
        adapt_prob(&mut probs.skip[i], &counts.skip[i]);
    }
    if interp_switchable {
        for i in 0..4 {
            adapt_probs(&INTERP_FILTER_TREE, &mut probs.interp_filter[i], &counts.interp_filter[i]);
        }
    }
    if tx_mode_select {
        for i in 0..2 {
            adapt_probs(&TX_SIZE_8_TREE, &mut probs.tx[1][i], &counts.tx[1][i]);
            adapt_probs(&TX_SIZE_16_TREE, &mut probs.tx[2][i], &counts.tx[2][i]);
            adapt_probs(&TX_SIZE_32_TREE, &mut probs.tx[3][i], &counts.tx[3][i]);
        }
    }
    adapt_probs(&MV_JOINT_TREE, &mut probs.mv_joint, &counts.mv_joint);
    for i in 0..2 {
        adapt_prob(&mut probs.mv_sign[i], &counts.mv_sign[i]);
        adapt_probs(&MV_CLASS_TREE, &mut probs.mv_class[i], &counts.mv_class[i]);
        adapt_prob(&mut probs.mv_class0_bit[i], &counts.mv_class0_bit[i]);
        for j in 0..10 {
            adapt_prob(&mut probs.mv_bits[i][j], &counts.mv_bits[i][j]);
        }
        for j in 0..2 {
            adapt_probs(&MV_FR_TREE, &mut probs.mv_class0_fr[i][j], &counts.mv_class0_fr[i][j]);
        }
        adapt_probs(&MV_FR_TREE, &mut probs.mv_fr[i], &counts.mv_fr[i]);
        if allow_hp {
            adapt_prob(&mut probs.mv_class0_hp[i], &counts.mv_class0_hp[i]);
            adapt_prob(&mut probs.mv_hp[i], &counts.mv_hp[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_prob_limits() {
        // No observations: unchanged.
        assert_eq!(merge_prob(77, 0, 0, 20, 128), 77);
        // Saturated all-zero observations pull halfway (factor 128) to 255.
        assert_eq!(merge_prob(1, 100, 0, 20, 128), ((256 - 128 + 255 * 128 + 128) >> 8) as u8);
        // All ones pull towards 1.
        assert!(merge_prob(200, 0, 50, 20, 128) < 110);
    }

    #[test]
    fn tree_adaptation_matches_binary_adaptation() {
        // A two-leaf tree adapts exactly like adapt_prob.
        let mut p = [90u8];
        adapt_probs(&BINARY_TREE, &mut p, &[13, 4]);
        let mut q = 90u8;
        adapt_prob(&mut q, &[13, 4]);
        assert_eq!(p[0], q);
    }
}
