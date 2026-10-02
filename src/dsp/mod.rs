//! The sample-level kernels: inverse transforms, intra and inter
//! prediction, loop filter. Scalar Rust, written to the specification's
//! formulas; the decoder above them never touches samples directly.

pub(crate) mod inter;
pub(crate) mod intra;
pub(crate) mod itx;
pub(crate) mod lf;
