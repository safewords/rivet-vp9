//! A VP9 decoder and encoder.
//!
//! Rust, no C, no system libraries, no build script. Written from the VP9
//! Bitstream & Decoding Process Specification (v0.6 / v0.7, Google and
//! Argon Design) — the syntax tables of its section 6, the decoding process
//! of section 8 and the tables of section 10 — and not translated from any
//! other implementation.
//!
//! # Decoding
//!
//! ```no_run
//! let mut dec = vp9::Decoder::new();
//! # let packets: Vec<Vec<u8>> = Vec::new();
//! for packet in &packets {
//!     if let Some(frame) = dec.decode(packet)? {
//!         // frame.plane(0), frame.plane(1), frame.plane(2): Y, U, V
//!     }
//! }
//! # Ok::<(), vp9::Error>(())
//! ```
//!
//! # Encoding
//!
//! ```
//! let mut enc = vp9::Encoder::new(vp9::Config::new(64, 64));
//! let frame = vp9::Frame::new(64, 64, 8, vp9::ChromaFormat::Yuv420);
//! let packet = enc.encode(&frame)?;
//! let decoded = vp9::Decoder::new().decode(&packet)?.unwrap();
//! assert_eq!((decoded.width, decoded.height), (64, 64));
//! # Ok::<(), vp9::Error>(())
//! ```
//!
//! # Layout
//!
//! - `header` — uncompressed and compressed frame headers (6.2, 6.3).
//! - `bool_coder` — the boolean decoder (9.2) and its inverse.
//! - `probs` — frame contexts, counts, backward adaptation (8.4).
//! - [`decoder`] — tiles, partitions, mode info, motion vector prediction,
//!   residual tokens, reconstruction, loop filter, reference slots.
//! - `dsp` — inverse transforms, intra and inter predictors, loop filter
//!   kernels.
//! - [`encoder`] — a key-frame and inter-frame encoder.
//! - [`ivf`] — the IVF container; [`superframe`] — Annex B superframes.

#![warn(missing_docs)]

pub(crate) mod bits;
pub(crate) mod bool_coder;
pub(crate) mod consts;
pub mod decoder;
pub(crate) mod dsp;
pub mod encoder;
pub mod frame;
pub(crate) mod header;
pub mod ivf;
pub(crate) mod probs;
pub mod superframe;
pub(crate) mod tables;

pub use decoder::Decoder;
pub use encoder::{Config, Encoder};
pub use frame::{ChromaFormat, ColorSpace, Frame, Plane};

/// Errors the decoder and encoder can report.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The bitstream is malformed: a syntax element out of range, data cut
    /// short, a reference to a frame that was never decoded.
    #[error("bitstream error: {0}")]
    Bitstream(String),
    /// The stream is valid but uses a feature this crate does not
    /// implement (yet). The message names it.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The caller's input is unusable: a frame of the wrong size or format,
    /// an invalid configuration, a file that is not IVF.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// `Result` with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    #[cold]
    #[inline(never)]
    pub(crate) fn bitstream(msg: impl Into<String>) -> Self {
        Error::Bitstream(msg.into())
    }
    #[cold]
    #[inline(never)]
    pub(crate) fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
    #[cold]
    #[inline(never)]
    pub(crate) fn invalid(msg: impl Into<String>) -> Self {
        Error::InvalidInput(msg.into())
    }
}
