//! Named constants, symbolic lookup tables and decode trees of the
//! specification (sections 3, 6, 7 and 9.3.1). The numeric tables are in
//! `tables.rs`; the ones here are written with the specification's names.

// Frame types (7.2).
pub(crate) const KEY_FRAME: u8 = 0;

// Block sizes (7.4.3).
pub(crate) const BLOCK_4X4: u8 = 0;
pub(crate) const BLOCK_4X8: u8 = 1;
pub(crate) const BLOCK_8X4: u8 = 2;
pub(crate) const BLOCK_8X8: u8 = 3;
pub(crate) const BLOCK_8X16: u8 = 4;
pub(crate) const BLOCK_16X8: u8 = 5;
pub(crate) const BLOCK_16X16: u8 = 6;
pub(crate) const BLOCK_16X32: u8 = 7;
pub(crate) const BLOCK_32X16: u8 = 8;
pub(crate) const BLOCK_32X32: u8 = 9;
pub(crate) const BLOCK_32X64: u8 = 10;
pub(crate) const BLOCK_64X32: u8 = 11;
pub(crate) const BLOCK_64X64: u8 = 12;
pub(crate) const BLOCK_INVALID: u8 = 14;

// Partition types.
pub(crate) const PARTITION_NONE: u8 = 0;
pub(crate) const PARTITION_HORZ: u8 = 1;
pub(crate) const PARTITION_VERT: u8 = 2;
pub(crate) const PARTITION_SPLIT: u8 = 3;

// Intra modes (7.4.5) and inter modes (7.4.11): one numbering, y_mode.
pub(crate) const DC_PRED: u8 = 0;
pub(crate) const V_PRED: u8 = 1;
pub(crate) const H_PRED: u8 = 2;
pub(crate) const D45_PRED: u8 = 3;
pub(crate) const D135_PRED: u8 = 4;
pub(crate) const D117_PRED: u8 = 5;
pub(crate) const D153_PRED: u8 = 6;
pub(crate) const D207_PRED: u8 = 7;
pub(crate) const D63_PRED: u8 = 8;
pub(crate) const TM_PRED: u8 = 9;
pub(crate) const NEARESTMV: u8 = 10;
pub(crate) const NEARMV: u8 = 11;
pub(crate) const ZEROMV: u8 = 12;
pub(crate) const NEWMV: u8 = 13;

// Transform sizes and modes (7.3.1, 7.4.8).
pub(crate) const TX_4X4: u8 = 0;
pub(crate) const TX_8X8: u8 = 1;
pub(crate) const TX_16X16: u8 = 2;
pub(crate) const TX_32X32: u8 = 3;
pub(crate) const ONLY_4X4: u8 = 0;
pub(crate) const ALLOW_32X32: u8 = 3;
pub(crate) const TX_MODE_SELECT: u8 = 4;

// Transform types (section 3).
pub(crate) const DCT_DCT: u8 = 0;
pub(crate) const ADST_DCT: u8 = 1;
pub(crate) const DCT_ADST: u8 = 2;
pub(crate) const ADST_ADST: u8 = 3;

// Reference frames (7.4.12). NONE sorts below INTRA_FRAME, which is all
// the specification relies on ("<= NONE", "> INTRA_FRAME").
pub(crate) const NONE: i8 = -1;
pub(crate) const INTRA_FRAME: i8 = 0;
pub(crate) const LAST_FRAME: i8 = 1;
pub(crate) const GOLDEN_FRAME: i8 = 2;
pub(crate) const ALTREF_FRAME: i8 = 3;

// Interpolation filters (7.2.7).
pub(crate) const EIGHTTAP_SMOOTH: u8 = 1;
pub(crate) const EIGHTTAP: u8 = 0;
pub(crate) const EIGHTTAP_SHARP: u8 = 2;
pub(crate) const BILINEAR: u8 = 3;
pub(crate) const SWITCHABLE: u8 = 4;

/// `literal_to_type` (6.2.7).
pub(crate) const LITERAL_TO_TYPE: [u8; 4] = [EIGHTTAP_SMOOTH, EIGHTTAP, EIGHTTAP_SHARP, BILINEAR];

// Reference modes (7.3.6).
pub(crate) const SINGLE_REFERENCE: u8 = 0;
pub(crate) const COMPOUND_REFERENCE: u8 = 1;
pub(crate) const REFERENCE_MODE_SELECT: u8 = 2;

// MV joints (7.4.13).
pub(crate) const MV_JOINT_ZERO: u8 = 0;
pub(crate) const MV_JOINT_HNZVZ: u8 = 1;
pub(crate) const MV_JOINT_HZVNZ: u8 = 2;
pub(crate) const MV_JOINT_HNZVNZ: u8 = 3;

// Tokens (7.4.16).
pub(crate) const ZERO_TOKEN: u8 = 0;
pub(crate) const DCT_VAL_CAT6: u8 = 10;

// Segmentation features.
pub(crate) const SEG_LVL_ALT_Q: usize = 0;
pub(crate) const SEG_LVL_ALT_L: usize = 1;
pub(crate) const SEG_LVL_REF_FRAME: usize = 2;
pub(crate) const SEG_LVL_SKIP: usize = 3;
pub(crate) const MAX_SEGMENTS: usize = 8;

// Motion vector constants (section 3).
pub(crate) const MV_BORDER: i32 = 128;
pub(crate) const INTERP_EXTEND: i32 = 4;
pub(crate) const BORDERINPIXELS: i32 = 160;
pub(crate) const COMPANDED_MVREF_THRESH: i32 = 8;
pub(crate) const MAX_LOOP_FILTER: i32 = 63;
pub(crate) const MIN_TILE_WIDTH_B64: u32 = 4;
pub(crate) const MAX_TILE_WIDTH_B64: u32 = 64;

// Inter mode contexts (section 3).
pub(crate) const BOTH_ZERO: u8 = 0;
pub(crate) const ZERO_PLUS_PREDICTED: u8 = 1;
pub(crate) const BOTH_PREDICTED: u8 = 2;
pub(crate) const NEW_PLUS_NON_INTRA: u8 = 3;
pub(crate) const BOTH_NEW: u8 = 4;
pub(crate) const INTRA_PLUS_NON_INTRA: u8 = 5;
pub(crate) const BOTH_INTRA: u8 = 6;
pub(crate) const INVALID_CASE: u8 = 9;

/// `counter_to_context` (6.5.1).
pub(crate) const COUNTER_TO_CONTEXT: [u8; 19] = [
    BOTH_PREDICTED,
    NEW_PLUS_NON_INTRA,
    BOTH_NEW,
    ZERO_PLUS_PREDICTED,
    NEW_PLUS_NON_INTRA,
    INVALID_CASE,
    BOTH_ZERO,
    INVALID_CASE,
    INVALID_CASE,
    INTRA_PLUS_NON_INTRA,
    INTRA_PLUS_NON_INTRA,
    INVALID_CASE,
    INTRA_PLUS_NON_INTRA,
    INVALID_CASE,
    INVALID_CASE,
    INVALID_CASE,
    INVALID_CASE,
    INVALID_CASE,
    BOTH_INTRA,
];

/// `max_txsize_lookup` (6.4.10).
pub(crate) const MAX_TXSIZE_LOOKUP: [u8; 13] = [
    TX_4X4, TX_4X4, TX_4X4, TX_8X8, TX_8X8, TX_8X8, TX_16X16, TX_16X16, TX_16X16, TX_32X32, TX_32X32,
    TX_32X32, TX_32X32,
];

/// `tx_mode_to_biggest_tx_size` (10.2).
pub(crate) const TX_MODE_TO_BIGGEST_TX_SIZE: [u8; 5] = [TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_32X32];

const I: u8 = BLOCK_INVALID;

/// `ss_size_lookup[bsize][subx][suby]` (6.4.23).
pub(crate) const SS_SIZE_LOOKUP: [[[u8; 2]; 2]; 13] = [
    [[BLOCK_4X4, I], [I, I]],
    [[BLOCK_4X8, BLOCK_4X4], [I, I]],
    [[BLOCK_8X4, I], [BLOCK_4X4, I]],
    [[BLOCK_8X8, BLOCK_8X4], [BLOCK_4X8, BLOCK_4X4]],
    [[BLOCK_8X16, BLOCK_8X8], [I, BLOCK_4X8]],
    [[BLOCK_16X8, I], [BLOCK_8X8, BLOCK_8X4]],
    [[BLOCK_16X16, BLOCK_16X8], [BLOCK_8X16, BLOCK_8X8]],
    [[BLOCK_16X32, BLOCK_16X16], [I, BLOCK_8X16]],
    [[BLOCK_32X16, I], [BLOCK_16X16, BLOCK_16X8]],
    [[BLOCK_32X32, BLOCK_32X16], [BLOCK_16X32, BLOCK_16X16]],
    [[BLOCK_32X64, BLOCK_32X32], [I, BLOCK_16X32]],
    [[BLOCK_64X32, I], [BLOCK_32X32, BLOCK_32X16]],
    [[BLOCK_64X64, BLOCK_64X32], [BLOCK_32X64, BLOCK_32X32]],
];

/// `subsize_lookup[partition][bsize]` (10.2).
pub(crate) const SUBSIZE_LOOKUP: [[u8; 13]; 4] = [
    [
        BLOCK_4X4,
        BLOCK_4X8,
        BLOCK_8X4,
        BLOCK_8X8,
        BLOCK_8X16,
        BLOCK_16X8,
        BLOCK_16X16,
        BLOCK_16X32,
        BLOCK_32X16,
        BLOCK_32X32,
        BLOCK_32X64,
        BLOCK_64X32,
        BLOCK_64X64,
    ],
    [I, I, I, BLOCK_8X4, I, I, BLOCK_16X8, I, I, BLOCK_32X16, I, I, BLOCK_64X32],
    [I, I, I, BLOCK_4X8, I, I, BLOCK_8X16, I, I, BLOCK_16X32, I, I, BLOCK_32X64],
    [I, I, I, BLOCK_4X4, I, I, BLOCK_8X8, I, I, BLOCK_16X16, I, I, BLOCK_32X32],
];

/// `mode2txfm_map` (10.2).
pub(crate) const MODE2TXFM_MAP: [u8; 14] = [
    DCT_DCT,   // DC
    ADST_DCT,  // V
    DCT_ADST,  // H
    DCT_DCT,   // D45
    ADST_ADST, // D135
    ADST_DCT,  // D117
    DCT_ADST,  // D153
    DCT_ADST,  // D207
    ADST_DCT,  // D63
    ADST_ADST, // TM
    DCT_DCT,   // NEARESTMV
    DCT_DCT,   // NEARMV
    DCT_DCT,   // ZEROMV
    DCT_DCT,   // NEWMV
];

/// `extra_bits[token]`: category, number of extra bits, base value (6.4.26).
pub(crate) const EXTRA_BITS: [[i32; 3]; 11] = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 2],
    [0, 0, 3],
    [0, 0, 4],
    [1, 1, 5],
    [2, 2, 7],
    [3, 3, 11],
    [4, 4, 19],
    [5, 5, 35],
    [6, 14, 67],
];

/// `cat_probs[cat]` (6.4.26), each row only as long as the category.
pub(crate) const CAT_PROBS: [&[u8]; 7] = [
    &[0],
    &[159],
    &[165, 145],
    &[173, 148, 140],
    &[176, 155, 140, 135],
    &[180, 157, 141, 134, 130],
    &[254, 254, 254, 252, 249, 243, 230, 196, 177, 153, 140, 133, 130, 129],
];

// Decode trees (9.3.1). Leaves are negated values; index 0 is never the
// target of a branch, so a leaf of value 0 is written as 0.
pub(crate) const PARTITION_TREE: [i8; 6] = [-(PARTITION_NONE as i8), 2, -(PARTITION_HORZ as i8), 4, -(PARTITION_VERT as i8), -(PARTITION_SPLIT as i8)];
pub(crate) const INTRA_MODE_TREE: [i8; 18] = [
    -(DC_PRED as i8),
    2,
    -(TM_PRED as i8),
    4,
    -(V_PRED as i8),
    6,
    8,
    12,
    -(H_PRED as i8),
    10,
    -(D135_PRED as i8),
    -(D117_PRED as i8),
    -(D45_PRED as i8),
    14,
    -(D63_PRED as i8),
    16,
    -(D153_PRED as i8),
    -(D207_PRED as i8),
];
pub(crate) const SEGMENT_TREE: [i8; 14] = [2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7];
pub(crate) const TX_SIZE_32_TREE: [i8; 6] = [-(TX_4X4 as i8), 2, -(TX_8X8 as i8), 4, -(TX_16X16 as i8), -(TX_32X32 as i8)];
pub(crate) const TX_SIZE_16_TREE: [i8; 4] = [-(TX_4X4 as i8), 2, -(TX_8X8 as i8), -(TX_16X16 as i8)];
pub(crate) const TX_SIZE_8_TREE: [i8; 2] = [-(TX_4X4 as i8), -(TX_8X8 as i8)];
/// inter_mode_tree; values are `inter_mode` = y_mode - NEARESTMV.
pub(crate) const INTER_MODE_TREE: [i8; 6] = [
    -2, // ZEROMV - NEARESTMV
    2,
    0, // NEARESTMV - NEARESTMV
    4,
    -1, // NEARMV - NEARESTMV
    -3, // NEWMV - NEARESTMV
];
pub(crate) const INTERP_FILTER_TREE: [i8; 4] = [-(EIGHTTAP as i8), 2, -(EIGHTTAP_SMOOTH as i8), -(EIGHTTAP_SHARP as i8)];
pub(crate) const MV_JOINT_TREE: [i8; 6] = [-(MV_JOINT_ZERO as i8), 2, -(MV_JOINT_HNZVZ as i8), 4, -(MV_JOINT_HZVNZ as i8), -(MV_JOINT_HNZVNZ as i8)];
pub(crate) const MV_CLASS_TREE: [i8; 20] = [0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10];
pub(crate) const MV_FR_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
pub(crate) const TOKEN_TREE: [i8; 20] = [0, 2, -1, 4, 6, 10, -2, 8, -3, -4, 12, 14, -5, -6, 16, 18, -7, -8, -9, -10];
/// small_token_tree (8.4.3), adapted from index 2.
pub(crate) const SMALL_TOKEN_TREE: [i8; 6] = [0, 0, 0, 4, -1, -2];
pub(crate) const BINARY_TREE: [i8; 2] = [0, -1];

#[cfg(test)]
mod table_tests {
    use crate::tables::*;

    fn is_perm(s: &[u16]) -> bool {
        let mut seen = vec![false; s.len()];
        for &v in s {
            if v as usize >= s.len() || seen[v as usize] {
                return false;
            }
            seen[v as usize] = true;
        }
        true
    }

    #[test]
    fn scans_are_permutations() {
        for s in [&DEFAULT_SCAN_4X4[..], &COL_SCAN_4X4, &ROW_SCAN_4X4] {
            assert!(is_perm(s));
        }
        for s in [&DEFAULT_SCAN_8X8[..], &COL_SCAN_8X8, &ROW_SCAN_8X8] {
            assert!(is_perm(s));
        }
        for s in [&DEFAULT_SCAN_16X16[..], &COL_SCAN_16X16, &ROW_SCAN_16X16] {
            assert!(is_perm(s));
        }
        assert!(is_perm(&DEFAULT_SCAN_32X32));
    }

    #[test]
    fn bands_are_monotone() {
        assert!(COEFBAND_8X8PLUS.windows(2).all(|w| w[0] <= w[1]));
        assert!(COEFBAND_4X4.windows(2).all(|w| w[0] <= w[1]));
    }
}
