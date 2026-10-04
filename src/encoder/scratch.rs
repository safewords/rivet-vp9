//! Thread-local free lists for the buffers the encoder's trials allocate
//! and drop by the thousand per frame (saved regions, coded transform
//! blocks): taken cleared, given back with their capacity.

use std::cell::RefCell;

use crate::decoder::MiInfo;

macro_rules! pool {
    ($name:ident, $take:ident, $give:ident, $t:ty) => {
        thread_local! {
            static $name: RefCell<Vec<Vec<$t>>> = const { RefCell::new(Vec::new()) };
        }

        /// An empty buffer, with the capacity of one given back if any.
        pub(crate) fn $take() -> Vec<$t> {
            $name.with(|p| p.borrow_mut().pop()).unwrap_or_default()
        }

        /// Keeps `v` for reuse.
        pub(crate) fn $give(mut v: Vec<$t>) {
            if v.capacity() == 0 {
                return;
            }
            v.clear();
            $name.with(|p| {
                let mut p = p.borrow_mut();
                if p.len() < 256 {
                    p.push(v);
                }
            });
        }
    };
}

pool!(SAMPLES, take_u16, give_u16, u16);
pool!(BYTES, take_u8, give_u8, u8);
pool!(COEFS, take_i32, give_i32, i32);
pool!(MODES, take_mi, give_mi, MiInfo);
pool!(
    BLOCKS,
    take_blocks,
    give_blocks,
    Option<(usize, usize, u8, usize)>
);
