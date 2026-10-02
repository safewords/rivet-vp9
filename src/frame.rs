//! The picture type: what the decoder hands back and the encoder takes.
//!
//! Its shape follows rivet-h26x's `Picture`: one buffer holding the planes
//! one after the other (Y, then U, then V), tightly packed, one byte per
//! sample at 8 bits and little-endian `u16` above — the layout of a raw
//! planar frame, and what the VP9 test vectors' per-frame MD5s hash.

/// Chroma sampling of a frame. VP9 profiles 0 and 2 are 4:2:0 only; profiles
/// 1 and 3 carry the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChromaFormat {
    /// 4:2:0 — chroma halved in both directions.
    Yuv420,
    /// 4:2:2 — chroma halved horizontally.
    Yuv422,
    /// 4:4:0 — chroma halved vertically.
    Yuv440,
    /// 4:4:4 — no subsampling.
    Yuv444,
}

impl ChromaFormat {
    /// `(SubWidthC, SubHeightC)` — how many luma samples one chroma sample
    /// covers in each direction.
    pub fn subsampling(self) -> (u32, u32) {
        let (x, y) = self.shifts();
        (1 << x, 1 << y)
    }

    /// The bitstream's `(subsampling_x, subsampling_y)`.
    pub fn shifts(self) -> (u32, u32) {
        match self {
            ChromaFormat::Yuv420 => (1, 1),
            ChromaFormat::Yuv422 => (1, 0),
            ChromaFormat::Yuv440 => (0, 1),
            ChromaFormat::Yuv444 => (0, 0),
        }
    }

    /// From the bitstream's `subsampling_x` and `subsampling_y`.
    pub fn from_shifts(ss_x: u32, ss_y: u32) -> Self {
        match (ss_x != 0, ss_y != 0) {
            (true, true) => ChromaFormat::Yuv420,
            (true, false) => ChromaFormat::Yuv422,
            (false, true) => ChromaFormat::Yuv440,
            (false, false) => ChromaFormat::Yuv444,
        }
    }
}

/// `color_space` of the uncompressed header (7.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorSpace {
    /// Not signalled in the bitstream.
    Unknown,
    /// Rec. ITU-R BT.601.
    #[default]
    Bt601,
    /// Rec. ITU-R BT.709.
    Bt709,
    /// SMPTE 170.
    Smpte170,
    /// SMPTE 240.
    Smpte240,
    /// Rec. ITU-R BT.2020.
    Bt2020,
    /// Reserved value 6.
    Reserved,
    /// sRGB: the planes are G, B, R (profiles 1 and 3 only).
    Rgb,
}

impl ColorSpace {
    pub(crate) fn from_bits(v: u32) -> Self {
        match v {
            0 => ColorSpace::Unknown,
            1 => ColorSpace::Bt601,
            2 => ColorSpace::Bt709,
            3 => ColorSpace::Smpte170,
            4 => ColorSpace::Smpte240,
            5 => ColorSpace::Bt2020,
            6 => ColorSpace::Reserved,
            _ => ColorSpace::Rgb,
        }
    }

    pub(crate) fn bits(self) -> u32 {
        match self {
            ColorSpace::Unknown => 0,
            ColorSpace::Bt601 => 1,
            ColorSpace::Bt709 => 2,
            ColorSpace::Smpte170 => 3,
            ColorSpace::Smpte240 => 4,
            ColorSpace::Bt2020 => 5,
            ColorSpace::Reserved => 6,
            ColorSpace::Rgb => 7,
        }
    }
}

/// One plane of a [`Frame`]: where it sits in the frame's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    /// Byte offset of the plane's first sample in [`Frame::data`].
    pub offset: usize,
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
}

/// A frame of planar YUV (or GBR, for [`ColorSpace::Rgb`]): one buffer
/// holding Y, then U, then V, each tightly packed (stride == width).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Luma width in samples.
    pub width: u32,
    /// Luma height in samples.
    pub height: u32,
    /// Bits per sample: 8, 10 or 12.
    pub bit_depth: u32,
    /// Chroma sampling.
    pub chroma: ChromaFormat,
    /// Colour space, as signalled.
    pub color_space: ColorSpace,
    /// `color_range`: true for full swing, false for studio swing.
    pub full_range: bool,
    /// The display size the stream asks for (`render_size`); a hint only.
    pub render_width: u32,
    /// See [`Self::render_width`].
    pub render_height: u32,
    /// The samples of every plane, packed: one byte each at 8 bits, else
    /// little-endian `u16` with the value in the low bits.
    pub data: Vec<u8>,
    /// Y, then U, then V.
    pub planes: Vec<Plane>,
}

impl Frame {
    /// A frame of the given geometry with every sample zero.
    pub fn new(width: u32, height: u32, bit_depth: u32, chroma: ChromaFormat) -> Self {
        let (sx, sy) = chroma.shifts();
        let bps = if bit_depth > 8 { 2 } else { 1 };
        let mut planes = Vec::with_capacity(3);
        let mut offset = 0usize;
        for i in 0..3 {
            let (w, h) = if i == 0 { (width, height) } else { ((width + sx) >> sx, (height + sy) >> sy) };
            planes.push(Plane { offset, width: w, height: h });
            offset += w as usize * h as usize * bps;
        }
        Frame {
            width,
            height,
            bit_depth,
            chroma,
            color_space: ColorSpace::Bt601,
            full_range: false,
            render_width: width,
            render_height: height,
            data: vec![0u8; offset],
            planes,
        }
    }

    /// Bytes per sample, the same for every plane: 1 at 8 bits, else 2.
    pub fn bytes_per_sample(&self) -> usize {
        if self.bit_depth > 8 { 2 } else { 1 }
    }

    fn plane_len(&self, i: usize) -> usize {
        let p = &self.planes[i];
        p.width as usize * p.height as usize * self.bytes_per_sample()
    }

    /// The bytes of plane `i` (0 Y, 1 U, 2 V).
    pub fn plane(&self, i: usize) -> &[u8] {
        let p = self.planes[i];
        &self.data[p.offset..p.offset + self.plane_len(i)]
    }

    /// The bytes of plane `i`, mutably.
    pub fn plane_mut(&mut self, i: usize) -> &mut [u8] {
        let p = self.planes[i];
        let len = self.plane_len(i);
        &mut self.data[p.offset..p.offset + len]
    }

    /// The sample at column `x`, row `y` of plane `i`.
    pub fn sample(&self, i: usize, x: u32, y: u32) -> u16 {
        let p = self.planes[i];
        let idx = y as usize * p.width as usize + x as usize;
        if self.bit_depth > 8 {
            let o = p.offset + 2 * idx;
            u16::from_le_bytes([self.data[o], self.data[o + 1]])
        } else {
            self.data[p.offset + idx] as u16
        }
    }

    /// Sets the sample at column `x`, row `y` of plane `i`.
    pub fn set_sample(&mut self, i: usize, x: u32, y: u32, v: u16) {
        let p = self.planes[i];
        let idx = y as usize * p.width as usize + x as usize;
        if self.bit_depth > 8 {
            let o = p.offset + 2 * idx;
            self.data[o..o + 2].copy_from_slice(&v.to_le_bytes());
        } else {
            self.data[p.offset + idx] = v as u8;
        }
    }

    /// The planes concatenated: Y then U then V.
    pub fn packed(&self) -> &[u8] {
        &self.data
    }

    /// The packed planes, taking the buffer.
    pub fn into_packed(self) -> Vec<u8> {
        self.data
    }
}
