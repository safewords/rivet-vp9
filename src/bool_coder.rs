//! The boolean (arithmetic) decoder of section 9.2, and the encoder that is
//! its exact inverse.
//!
//! The decoder keeps the specification's `BoolValue` in the top byte of a
//! 64-bit window whose lower bits are the stream bits still to be shifted
//! in, so normalisation is a shift instead of a bit-at-a-time loop. Bits past
//! the end of the data read as zero, as `newBit` does once `BoolMaxBits` is
//! exhausted.

use crate::{Error, Result};

pub(crate) struct BoolDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    /// `BoolValue` in bits 63..56, then the following stream bits.
    value: u64,
    /// Number of valid stream bits in `value` (may go negative past the end;
    /// the missing bits are zero).
    bits: i32,
    range: u32,
}

impl<'a> BoolDecoder<'a> {
    /// init_bool( sz ) over `data` (whose length is sz).
    pub(crate) fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::bitstream("boolean-coded partition of size 0"));
        }
        let mut d = BoolDecoder {
            data,
            pos: 0,
            value: 0,
            bits: 0,
            range: 255,
        };
        d.fill();
        if d.read(128) {
            return Err(Error::bitstream("boolean decoder marker bit is not 0"));
        }
        Ok(d)
    }

    #[inline(always)]
    fn fill(&mut self) {
        if self.bits >= 0 && self.pos + 8 <= self.data.len() {
            // As many whole bytes as fit, from one 8-byte load: the loop
            // below's bytes, at once.
            let bits = self.bits as u32;
            let k = (64 - bits) / 8;
            let keep = bits + 8 * k;
            let x = u64::from_be_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
            let mask = if keep >= 64 { !0 } else { !0u64 << (64 - keep) };
            self.value |= (x >> bits) & mask;
            self.pos += k as usize;
            self.bits += 8 * k as i32;
            return;
        }
        while self.bits <= 56 {
            if self.pos < self.data.len() {
                self.value |= (self.data[self.pos] as u64) << (56 - self.bits);
                self.pos += 1;
                self.bits += 8;
            } else {
                // Zeros from here on; pretend they were loaded.
                self.bits = 64;
                break;
            }
        }
    }

    /// read_bool( p ): a bool whose probability of being 0 is p/256.
    #[inline(always)]
    pub(crate) fn read(&mut self, p: u8) -> bool {
        let split = 1 + (((self.range - 1) * p as u32) >> 8);
        let big = (split as u64) << 56;
        let bit = if self.value < big {
            self.range = split;
            false
        } else {
            self.range -= split;
            self.value -= big;
            true
        };
        let shift = self.range.leading_zeros() - 24;
        self.range <<= shift;
        self.value <<= shift;
        self.bits -= shift as i32;
        if self.bits < 16 {
            self.fill();
        }
        bit
    }

    /// Whether the bits not yet read — the padding exit_bool reads (9.2.3)
    /// — are all zero, as bitstream conformance requires. A tile decoded
    /// with the wrong syntax leaves coded data unread.
    pub(crate) fn padding_is_zero(&self) -> bool {
        // The top byte of `value` is BoolValue, already read; below it,
        // stream bits not yet read.
        self.value & ((1u64 << 56) - 1) == 0 && self.data[self.pos..].iter().all(|&b| b == 0)
    }

    /// read_literal( n ) (9.2.4).
    pub(crate) fn literal(&mut self, n: u32) -> u32 {
        let mut x = 0;
        for _ in 0..n {
            x = (x << 1) | self.read(128) as u32;
        }
        x
    }

    /// Decodes a tree-coded value (9.3.3) with probabilities from `prob`,
    /// called with the node index (`n >> 1`).
    #[inline(always)]
    pub(crate) fn tree(&mut self, tree: &[i8], mut prob: impl FnMut(usize) -> u8) -> u8 {
        let mut n: i32 = 0;
        loop {
            let p = prob((n >> 1) as usize);
            n = tree[n as usize + self.read(p) as usize] as i32;
            if n <= 0 {
                return (-n) as u8;
            }
        }
    }
}

/// Where the encoder writes bools: the [`BoolEncoder`], or a [`BitCounter`]
/// that only adds up what they would cost (the encoder's trial codings).
pub(crate) trait Sink {
    /// Writes `bit` with probability `p` (of a 0, out of 256).
    fn write(&mut self, bit: bool, p: u8);

    /// An `n`-bit literal, most significant bit first.
    fn literal(&mut self, n: u32, v: u32) {
        for i in (0..n).rev() {
            self.write((v >> i) & 1 != 0, 128);
        }
    }

    /// Encodes `value` with `tree`; `prob` gives the probability of each
    /// node index visited.
    fn tree(&mut self, tree: &[i8], value: u8, mut prob: impl FnMut(usize) -> u8)
    where
        Self: Sized,
    {
        // Find the path from the root to the leaf by search.
        fn path(
            tree: &[i8],
            node: usize,
            value: u8,
            out: &mut [(usize, bool)],
            n: &mut usize,
        ) -> bool {
            for b in 0..2 {
                let t = tree[node + b];
                out[*n] = (node, b == 1);
                *n += 1;
                if t <= 0 {
                    if (-t) as u8 == value {
                        return true;
                    }
                } else if path(tree, t as usize, value, out, n) {
                    return true;
                }
                *n -= 1;
            }
            false
        }
        let mut p = [(0usize, false); 16];
        let mut n = 0;
        let found = path(tree, 0, value, &mut p, &mut n);
        debug_assert!(found, "value not in tree");
        for &(node, bit) in &p[..n] {
            self.write(bit, prob(node >> 1));
        }
    }
}

/// The cost in bits of a bool of probability `p` (of a 0): `[cost of 0,
/// cost of 1]`.
pub(crate) fn bool_cost(p: u8) -> [f64; 2] {
    static TABLE: std::sync::OnceLock<[[f64; 2]; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [[0.0; 2]; 256];
        for (p, c) in t.iter_mut().enumerate() {
            let p0 = (p.max(1) as f64) / 256.0;
            *c = [-p0.log2(), -(1.0 - p0).log2()];
        }
        t
    })[p as usize]
}

/// A [`Sink`] that counts the bits a [`BoolEncoder`] would spend.
#[derive(Default, Debug, Clone, Copy)]
pub(crate) struct BitCounter {
    pub bits: f64,
}

impl Sink for BitCounter {
    #[inline]
    fn write(&mut self, bit: bool, p: u8) {
        self.bits += bool_cost(p)[bit as usize];
    }
}

impl Sink for BoolEncoder {
    #[inline]
    fn write(&mut self, bit: bool, p: u8) {
        BoolEncoder::write(self, bit, p);
    }
}

/// The boolean encoder: produces a stream [`BoolDecoder`] reads back.
///
/// It tracks the low end of the coding interval at the decoder's scale:
/// bits 0..8 of `low` are aligned with the decoder's `BoolValue` window,
/// bits above it are stream bits already passed over (`count` of them not
/// yet written out). A carry out of `low` ripples into the bytes written.
pub(crate) struct BoolEncoder {
    buf: Vec<u8>,
    low: u64,
    range: u32,
    count: u32,
}

impl Default for BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolEncoder {
    pub(crate) fn new() -> Self {
        let mut e = BoolEncoder {
            buf: Vec::new(),
            low: 0,
            range: 255,
            count: 0,
        };
        // The marker bit init_bool reads.
        e.write(false, 128);
        e
    }

    fn carry(&mut self) {
        for b in self.buf.iter_mut().rev() {
            if *b == 0xff {
                *b = 0;
            } else {
                *b += 1;
                return;
            }
        }
    }

    pub(crate) fn write(&mut self, bit: bool, p: u8) {
        let split = 1 + (((self.range - 1) * p as u32) >> 8);
        if bit {
            self.low += split as u64;
            self.range -= split;
        } else {
            self.range = split;
        }
        let shift = self.range.leading_zeros() - 24;
        self.range <<= shift;
        self.low <<= shift;
        self.count += shift;
        let top = 8 + self.count;
        if self.low >> top != 0 {
            self.carry();
            self.low &= (1u64 << top) - 1;
        }
        while self.count >= 8 {
            let byte = (self.low >> self.count) as u8;
            self.buf.push(byte);
            self.count -= 8;
            self.low &= (1u64 << (8 + self.count)) - 1;
        }
    }

    /// Flushes the interval: the stream is `low` followed by zeros. Pads so
    /// the last byte is not taken for a superframe marker (9.2.3).
    pub(crate) fn finish(mut self) -> Vec<u8> {
        let total = 8 + self.count;
        let nbytes = total.div_ceil(8);
        let v = self.low << (nbytes * 8 - total);
        for i in (0..nbytes).rev() {
            self.buf.push((v >> (8 * i)) as u8);
        }
        if self.buf.last().is_some_and(|&b| b & 0xe0 == 0xc0) {
            self.buf.push(0);
        }
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_check() {
        // What the encoder writes ends in zero padding; data left unread
        // is not padding.
        let mut e = BoolEncoder::new();
        for i in 0..200u32 {
            e.write(i % 3 == 0, (i * 37 % 255 + 1) as u8);
        }
        let mut data = e.finish();
        let read = |data: &[u8]| {
            let mut d = BoolDecoder::new(data).unwrap();
            for i in 0..200u32 {
                assert_eq!(d.read((i * 37 % 255 + 1) as u8), i % 3 == 0);
            }
            d.padding_is_zero()
        };
        assert!(read(&data));
        data.extend_from_slice(&[0, 0, 0]);
        assert!(read(&data));
        data.push(0x40);
        assert!(!read(&data));
    }

    /// The decoder exactly as section 9.2 writes it, bit by bit.
    struct SpecDecoder<'a> {
        data: &'a [u8],
        pos: usize,
        value: u32,
        range: u32,
        max_bits: i64,
    }

    impl<'a> SpecDecoder<'a> {
        fn read_bit(&mut self) -> u32 {
            let b = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1;
            self.pos += 1;
            b as u32
        }
        fn new(data: &'a [u8]) -> Self {
            let mut d = SpecDecoder {
                data,
                pos: 0,
                value: 0,
                range: 255,
                max_bits: 8 * data.len() as i64 - 8,
            };
            for _ in 0..8 {
                d.value = 2 * d.value + d.read_bit();
            }
            assert!(!d.read(128));
            d
        }
        fn read(&mut self, p: u8) -> bool {
            let split = 1 + (((self.range - 1) * p as u32) >> 8);
            let bit = if self.value < split {
                self.range = split;
                false
            } else {
                self.range -= split;
                self.value -= split;
                true
            };
            while self.range < 128 {
                let nb = if self.max_bits > 0 {
                    self.max_bits -= 1;
                    self.read_bit()
                } else {
                    0
                };
                self.range *= 2;
                self.value = (self.value << 1) + nb;
            }
            bit
        }
    }

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    #[test]
    fn encoder_decoder_round_trip_matches_spec_decoder() {
        let mut seed = 7u64;
        for trial in 0..200 {
            let n = 1 + (lcg(&mut seed) % 3000) as usize;
            let mut syms = Vec::with_capacity(n);
            for _ in 0..n {
                let p = 1 + (lcg(&mut seed) % 255) as u8;
                // Skew the bits towards the probability sometimes, against it others.
                let r = lcg(&mut seed) % 256;
                let bit = if trial % 3 == 0 {
                    r % 2 == 1
                } else {
                    r >= p as u32
                };
                syms.push((p, bit));
            }
            let mut e = BoolEncoder::new();
            for &(p, b) in &syms {
                e.write(b, p);
            }
            let data = e.finish();
            let mut fast = BoolDecoder::new(&data).unwrap();
            let mut spec = SpecDecoder::new(&data);
            for (i, &(p, b)) in syms.iter().enumerate() {
                let s = spec.read(p);
                assert_eq!(s, b, "spec decoder, trial {trial} symbol {i}");
                assert_eq!(fast.read(p), b, "fast decoder, trial {trial} symbol {i}");
            }
        }
    }

    #[test]
    fn literal_and_tree() {
        let mut e = BoolEncoder::new();
        e.literal(7, 93);
        for v in 0..10u8 {
            e.tree(&crate::consts::INTRA_MODE_TREE, v, |n| 30 + 20 * n as u8);
        }
        let data = e.finish();
        let mut d = BoolDecoder::new(&data).unwrap();
        assert_eq!(d.literal(7), 93);
        for v in 0..10u8 {
            assert_eq!(
                d.tree(&crate::consts::INTRA_MODE_TREE, |n| 30 + 20 * n as u8),
                v
            );
        }
    }
}
