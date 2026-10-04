#!/usr/bin/env python3
"""Regenerates src/tables.rs from the VP9 specification.

Usage: tools/gen_tables.py vp9-bitstream-specification-v0.7-20170222-draft.pdf

The PDF is the public "VP9 Bitstream & Decoding Process Specification",
v0.7 (22 February 2017), from
https://storage.googleapis.com/downloads.webmproject.org/docs/vp9/vp9-bitstream-specification-v0.7-20170222-draft.pdf
(v0.6, 31 March 2016, has the same tables). Needs pdftotext (poppler or
xpdf). Every `name[ dims ] = { ... }` in the text is read, page headers,
footers and comments dropped, and the numeric tables written out as Rust
statics, checked against their declared sizes. The symbolic tables (block
sizes, transform types by name) are written by hand in src/consts.rs.
"""
import json, re, subprocess, sys

def extract(pdf):
    text = subprocess.run(['pdftotext', '-simple', pdf, '-'], capture_output=True, check=True).stdout.decode('latin-1')
    clean = []
    for ln in text.replace('\r', '').split('\n'):
        if 'Copyright' in ln:
            continue
        if 'VP9 Bitstream & Decoding Process Specification' in ln:
            continue
        s = ln.strip()
        if re.fullmatch(r'\d{1,3}', s) and len(ln) - len(ln.lstrip()) > 40:
            continue  # page number
        clean.append(ln)
    text = '\n'.join(clean)
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    text = re.sub(r'//[^\n]*', ' ', text)
    out = {}
    for m in re.finditer(r'([a-z][a-z0-9_]*)((?:\s*\[[^\]]*\])+)\s*=\s*\{', text):
        i = m.end(); depth = 1
        while depth:
            c = text[i]
            if c == '{': depth += 1
            elif c == '}': depth -= 1
            i += 1
        body = text[m.end():i-1]
        out[m.group(1)] = re.findall(r'-?\d+|[A-Z][A-Z0-9_]+', body)
    return out

spec = [
 ('inv_map_table','INV_MAP_TABLE','u8',[255],'`inv_map_table` (6.3.5).'),
 ('dc_qlookup','DC_QLOOKUP','i32',[3,256],'`dc_qlookup` (8.6.1), indexed by `(BitDepth - 8) >> 1`.'),
 ('ac_qlookup','AC_QLOOKUP','i32',[3,256],'`ac_qlookup` (8.6.1).'),
 ('subpel_filters','SUBPEL_FILTERS','i32',[4,16,8],'`subpel_filters` (8.5.2.4), indexed by interp_filter (0 regular, 1 smooth, 2 sharp, 3 bilinear).'),
 ('default_scan_4x4','DEFAULT_SCAN_4X4','u16',[16],'10.1.'),
 ('col_scan_4x4','COL_SCAN_4X4','u16',[16],'10.1.'),
 ('row_scan_4x4','ROW_SCAN_4X4','u16',[16],'10.1.'),
 ('default_scan_8x8','DEFAULT_SCAN_8X8','u16',[64],'10.1.'),
 ('col_scan_8x8','COL_SCAN_8X8','u16',[64],'10.1.'),
 ('row_scan_8x8','ROW_SCAN_8X8','u16',[64],'10.1.'),
 ('default_scan_16x16','DEFAULT_SCAN_16X16','u16',[256],'10.1.'),
 ('col_scan_16x16','COL_SCAN_16X16','u16',[256],'10.1.'),
 ('row_scan_16x16','ROW_SCAN_16X16','u16',[256],'10.1.'),
 ('default_scan_32x32','DEFAULT_SCAN_32X32','u16',[1024],'10.1.'),
 ('coefband_4x4','COEFBAND_4X4','u8',[16],'10.2.'),
 ('coefband_8x8plus','COEFBAND_8X8PLUS','u8',[1024],'10.2.'),
 ('energy_class','ENERGY_CLASS','u8',[12],'10.2.'),
 ('pareto_table','PARETO_TABLE','u8',[128,8],'10.3.'),
 ('kf_partition_probs','KF_PARTITION_PROBS','u8',[16,3],'10.4.'),
 ('kf_y_mode_probs','KF_Y_MODE_PROBS','u8',[10,10,9],'10.4, `[abovemode][leftmode][node]`.'),
 ('kf_uv_mode_probs','KF_UV_MODE_PROBS','u8',[10,9],'10.4.'),
 ('default_partition_probs','DEFAULT_PARTITION_PROBS','u8',[16,3],'10.5.'),
 ('default_y_mode_probs','DEFAULT_Y_MODE_PROBS','u8',[4,9],'10.5.'),
 ('default_uv_mode_probs','DEFAULT_UV_MODE_PROBS','u8',[10,9],'10.5.'),
 ('default_skip_prob','DEFAULT_SKIP_PROB','u8',[3],'10.5.'),
 ('default_is_inter_prob','DEFAULT_IS_INTER_PROB','u8',[4],'10.5.'),
 ('default_comp_mode_prob','DEFAULT_COMP_MODE_PROB','u8',[5],'10.5.'),
 ('default_comp_ref_prob','DEFAULT_COMP_REF_PROB','u8',[5],'10.5.'),
 ('default_single_ref_prob','DEFAULT_SINGLE_REF_PROB','u8',[5,2],'10.5.'),
 ('default_mv_sign_prob','DEFAULT_MV_SIGN_PROB','u8',[2],'10.5.'),
 ('default_mv_bits_prob','DEFAULT_MV_BITS_PROB','u8',[2,10],'10.5.'),
 ('default_mv_class0_bit_prob','DEFAULT_MV_CLASS0_BIT_PROB','u8',[2],'10.5.'),
 ('default_tx_probs','DEFAULT_TX_PROBS','u8',[4,2,3],'10.5, `[maxTxSize][ctx][node]` (row 0 unused).'),
 ('default_inter_mode_probs','DEFAULT_INTER_MODE_PROBS','u8',[7,3],'10.5.'),
 ('default_interp_filter_probs','DEFAULT_INTERP_FILTER_PROBS','u8',[4,2],'10.5.'),
 ('default_mv_joint_probs','DEFAULT_MV_JOINT_PROBS','u8',[3],'10.5.'),
 ('default_mv_class_probs','DEFAULT_MV_CLASS_PROBS','u8',[2,10],'10.5.'),
 ('default_mv_class0_fr_probs','DEFAULT_MV_CLASS0_FR_PROBS','u8',[2,2,3],'10.5.'),
 ('default_mv_class0_hp_prob','DEFAULT_MV_CLASS0_HP_PROB','u8',[2],'10.5.'),
 ('default_mv_fr_probs','DEFAULT_MV_FR_PROBS','u8',[2,3],'10.5.'),
 ('default_mv_hp_prob','DEFAULT_MV_HP_PROB','u8',[2],'10.5.'),
 ('default_coef_probs','DEFAULT_COEF_PROBS','u8',[4,2,2,6,6,3],'10.5, `[txSz][plane>0][is_inter][band][ctx][node]` (band 0 has 3 contexts; the rest of its rows are zero).'),
 ('mv_ref_blocks','MV_REF_BLOCKS','i8',[13,8,2],'`mv_ref_blocks` (6.5.1): `[MiSize][i][row, col]` candidate offsets.'),
 ('mode_2_counter','MODE_2_COUNTER','u8',[14],'`mode_2_counter` (6.5.1).'),
 ('idx_n_column_to_subblock','IDX_N_COLUMN_TO_SUBBLOCK','u8',[4,2],'6.5.11.'),
 ('b_width_log2_lookup','B_WIDTH_LOG2','u8',[13],'10.2.'),
 ('b_height_log2_lookup','B_HEIGHT_LOG2','u8',[13],'10.2.'),
 ('num_4x4_blocks_wide_lookup','NUM_4X4_WIDE','u8',[13],'10.2.'),
 ('num_4x4_blocks_high_lookup','NUM_4X4_HIGH','u8',[13],'10.2.'),
 ('mi_width_log2_lookup','MI_WIDTH_LOG2','u8',[13],'10.2.'),
 ('num_8x8_blocks_wide_lookup','NUM_8X8_WIDE','u8',[13],'10.2.'),
 ('num_8x8_blocks_high_lookup','NUM_8X8_HIGH','u8',[13],'10.2.'),
 ('size_group_lookup','SIZE_GROUP','u8',[13],'10.2.'),
 ('segmentation_feature_bits','SEGMENTATION_FEATURE_BITS','u8',[4],'6.2.11.'),
 ('segmentation_feature_signed','SEGMENTATION_FEATURE_SIGNED','u8',[4],'6.2.11.'),
]

def main():
    T = extract(sys.argv[1])
    out = []
    for name, rust, ty, dims, doc in spec:
        vals = [int(v) for v in T[name]]
        total = 1
        for d in dims:
            total *= d
        assert len(vals) == total, (name, len(vals), total)
        def build(d, vs):
            if len(d) == 1:
                return '[' + ', '.join(str(v) for v in vs) + ']'
            step = len(vs) // d[0]
            return '[' + ', '.join(build(d[1:], vs[i*step:(i+1)*step]) for i in range(d[0])) + ']'
        t = ty
        for d in reversed(dims):
            t = f'[{t}; {d}]'
        if name == 'default_coef_probs':
            t = 'CoefProbs'
        out.append(f'/// {doc}\npub(crate) static {rust}: {t} = {build(dims, vals)};\n')
    hdr = '''//! Constant tables of the VP9 specification.
//!
//! Transcribed from the VP9 Bitstream & Decoding Process Specification
//! v0.7 (22 February 2017, Google / Argon Design), sections 6, 8 and 10, by
//! `tools/gen_tables.py` from the text of the published PDF: probability
//! tables, scans, quantiser lookups, filter taps and block-size lookups. They
//! are data, reproduced from the specification; the section each comes from
//! is noted on it. Do not edit by hand: regenerate.

#![allow(clippy::unreadable_literal)]

/// `[txSz][plane > 0][is_inter][band][ctx][node]`.
pub(crate) type CoefProbs = [[[[[[u8; 3]; 6]; 6]; 2]; 2]; 4];

'''
    sys.stdout.write(hdr + '\n'.join(out))

main()
