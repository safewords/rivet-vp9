//! The plain bit reader and writer of the uncompressed header: f(n) and
//! s(n) (sections 4.9.1, 4.9.2 and 9.1), most significant bit first.

use crate::{Error, Result};

/// Reads f(n) fields from a byte slice.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize, // in bits
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    /// The bitstream position indicator, in bits.
    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    pub(crate) fn bit(&mut self) -> Result<u32> {
        let byte = self.pos >> 3;
        if byte >= self.data.len() {
            return Err(Error::bitstream(
                "uncompressed header runs past the end of the frame",
            ));
        }
        let b = (self.data[byte] >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(b as u32)
    }

    /// f(n).
    pub(crate) fn f(&mut self, n: u32) -> Result<u32> {
        let mut x = 0;
        for _ in 0..n {
            x = (x << 1) | self.bit()?;
        }
        Ok(x)
    }

    pub(crate) fn flag(&mut self) -> Result<bool> {
        Ok(self.bit()? != 0)
    }

    /// s(n): magnitude then sign.
    pub(crate) fn s(&mut self, n: u32) -> Result<i32> {
        let v = self.f(n)? as i32;
        Ok(if self.flag()? { -v } else { v })
    }

    /// trailing_bits(): advance to a byte boundary.
    pub(crate) fn byte_align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }
}

/// Writes f(n) fields; the encoder's counterpart of [`BitReader`].
#[derive(Default)]
pub(crate) struct BitWriter {
    pub(crate) buf: Vec<u8>,
    nbits: usize,
}

impl BitWriter {
    pub(crate) fn bit(&mut self, b: bool) {
        if self.nbits & 7 == 0 {
            self.buf.push(0);
        }
        if b {
            *self.buf.last_mut().unwrap() |= 0x80 >> (self.nbits & 7);
        }
        self.nbits += 1;
    }

    pub(crate) fn f(&mut self, n: u32, v: u32) {
        for i in (0..n).rev() {
            self.bit((v >> i) & 1 != 0);
        }
    }

    /// Byte position after padding to a byte boundary with zero bits.
    pub(crate) fn finish(self) -> Vec<u8> {
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut w = BitWriter::default();
        w.f(2, 2);
        w.f(1, 1);
        w.f(6, 17);
        w.f(1, 1);
        w.f(16, 0xbeef);
        w.f(4, 5);
        w.f(1, 0);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.f(2).unwrap(), 2);
        assert_eq!(r.f(1).unwrap(), 1);
        assert_eq!(r.s(6).unwrap(), -17);
        assert_eq!(r.f(16).unwrap(), 0xbeef);
        assert_eq!(r.s(4).unwrap(), 5);
        r.byte_align();
        assert_eq!(r.position(), bytes.len() * 8);
    }
}
