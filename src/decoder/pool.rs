//! Reuse of the large per-frame buffers: the planes of a frame and of its
//! tile columns, and the mode info. Allocating and freeing megabytes per
//! frame costs more than decoding a fast frame (the allocator returns them
//! to the system, and the next frame faults their pages back in), so freed
//! buffers are kept here, up to a total size, and handed out again
//! zero-filled: a buffer from the pool is indistinguishable from a new one.

use std::sync::Mutex;

use super::MiInfo;

/// The most memory the pools keep, in bytes (each).
const KEEP_BYTES: usize = 256 << 20;

struct Pool<T> {
    free: Mutex<(Vec<Vec<T>>, usize)>,
}

impl<T: Clone> Pool<T> {
    const fn new() -> Self {
        Pool {
            free: Mutex::new((Vec::new(), 0)),
        }
    }

    /// A buffer of `len` copies of `fill`: the smallest kept one large
    /// enough, or a new one.
    fn take(&self, len: usize, fill: T) -> Vec<T> {
        let mut v = {
            let mut g = self.free.lock().unwrap_or_else(|e| e.into_inner());
            let best =
                g.0.iter()
                    .enumerate()
                    .filter(|(_, v)| v.capacity() >= len)
                    .min_by_key(|(_, v)| v.capacity())
                    .map(|(i, _)| i);
            match best {
                Some(i) => {
                    let v = g.0.swap_remove(i);
                    g.1 -= v.capacity() * size_of::<T>();
                    v
                }
                None => Vec::new(),
            }
        };
        v.clear();
        v.resize(len, fill);
        v
    }

    /// Keeps `v` for reuse, if there is room.
    fn give(&self, v: Vec<T>) {
        let bytes = v.capacity() * size_of::<T>();
        if bytes < 64 << 10 {
            return;
        }
        let mut g = self.free.lock().unwrap_or_else(|e| e.into_inner());
        if g.1 + bytes <= KEEP_BYTES {
            g.1 += bytes;
            g.0.push(v);
        }
    }
}

static SAMPLES: Pool<u16> = Pool::new();
static MODES: Pool<MiInfo> = Pool::new();

/// `len` zero samples.
pub(crate) fn samples(len: usize) -> Vec<u16> {
    SAMPLES.take(len, 0)
}

/// Returns a sample buffer for reuse.
pub(crate) fn give_samples(v: Vec<u16>) {
    SAMPLES.give(v);
}

/// `len` default mode infos.
pub(crate) fn modes(len: usize) -> Vec<MiInfo> {
    MODES.take(len, MiInfo::default())
}

/// Returns a mode info buffer for reuse.
pub(crate) fn give_modes(v: Vec<MiInfo>) {
    MODES.give(v);
}
