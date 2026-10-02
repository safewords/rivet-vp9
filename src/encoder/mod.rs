//! The encoder (placeholder until written).

use crate::{Error, Frame, Result};

/// Encoder configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Config {
    /// A configuration for `width` x `height`.
    pub fn new(width: u32, height: u32) -> Self {
        Config { width, height }
    }
}

/// A VP9 encoder.
pub struct Encoder {
    cfg: Config,
}

impl Encoder {
    /// A new encoder.
    pub fn new(cfg: Config) -> Result<Self> {
        Ok(Encoder { cfg })
    }

    /// Encodes one frame.
    pub fn encode(&mut self, _frame: &Frame) -> Result<Vec<u8>> {
        let _ = &self.cfg;
        Err(Error::invalid("not yet"))
    }
}
