//! Kernel timings, scalar against the SIMD level of the machine:
//! `cargo test --release --lib kernel_bench -- --ignored --nocapture`.
//! Each figure is the best of 15 batches, in nanoseconds per call.

use super::{Level, Rng, detected, inter, itx, lf, pixel};
use std::hint::black_box;
use std::time::Instant;

fn best(mut f: impl FnMut()) -> f64 {
    let iters = 2000;
    let mut best = f64::MAX;
    for _ in 0..15 {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        best = best.min(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    best
}

fn row(name: &str, s: f64, v: f64) {
    println!("| {name:<34} | {s:>9.1} | {v:>9.1} | {:>5.1}x |", s / v);
}

#[test]
#[ignore]
fn kernel_bench() {
    let simd = detected();
    let mut rng = Rng(1);
    println!("| kernel (ns/call) | scalar | {simd:?} | speedup |");
    println!("|---|---:|---:|---:|");
    // Inverse transforms, dense coefficients, 8-bit.
    for n in 2..=5u32 {
        let n0 = 1usize << n;
        let c: Vec<i32> = (0..n0 * n0).map(|_| rng.range(-300, 300) as i32).collect();
        let mut d = vec![128u16; n0 * n0];
        let f = |l: Level, d: &mut Vec<u16>| {
            itx::inverse_transform_add(l, black_box(&c), n, 0, false, 9, 8, d, n0)
        };
        let s = best(|| f(Level::Scalar, &mut d));
        let v = best(|| f(simd, &mut d));
        row(&format!("idct {n0}x{n0} + add"), s, v);
    }
    // Inter prediction, both passes.
    let plane: Vec<u16> = (0..256 * 256).map(|_| rng.range(0, 255) as u16).collect();
    let r = inter::RefPlane {
        data: &plane,
        stride: 256,
        last_x: 255,
        last_y: 255,
    };
    let mut sc = inter::Scratch::new();
    for w in [8usize, 16, 64] {
        let mut out = vec![0u16; w * w];
        let mut f = |l: Level| {
            inter::predict(
                l,
                &mut sc,
                &r,
                50 * 16 + 5,
                60 * 16 + 9,
                16,
                16,
                w,
                w,
                0,
                8,
                &mut out,
                w,
            )
        };
        let s = best(|| f(Level::Scalar));
        let v = best(|| f(simd));
        row(&format!("8-tap h+v {w}x{w}"), s, v);
    }
    // Loop filter: one superblock pass, every run filtered.
    let pic: Vec<u16> = (0..96 * 96).map(|i| 100 + ((i / 7) % 3) as u16).collect();
    let mut e = lf::Edges::new();
    for ed in 0..16 {
        for rr in 0..16 {
            e.fs[ed][rr] = if ed % 4 == 0 { 2 } else { 1 };
            e.limit[ed][rr] = 5;
            e.blimit[ed][rr] = 40;
            e.thresh[ed][rr] = 2;
        }
    }
    for vertical in [true, false] {
        let mut b = pic.clone();
        let s = best(|| {
            lf::filter_edges(
                Level::Scalar,
                &mut b,
                96,
                16,
                16,
                vertical,
                0,
                16,
                16,
                &e,
                8,
            )
        });
        let v = best(|| lf::filter_edges(simd, &mut b, 96, 16, 16, vertical, 0, 16, 16, &e, 8));
        row(
            &format!(
                "loop filter 64x64 {}",
                if vertical { "vertical" } else { "horizontal" }
            ),
            s,
            v,
        );
    }
    // Distortion.
    let a: Vec<u16> = (0..64 * 64).map(|_| rng.range(0, 255) as u16).collect();
    let b: Vec<u16> = (0..64 * 64).map(|_| rng.range(0, 255) as u16).collect();
    for w in [8usize, 16, 64] {
        let s = best(|| {
            black_box(pixel::sad(Level::Scalar, &a, 64, &b, 64, w, w));
        });
        let v = best(|| {
            black_box(pixel::sad(simd, &a, 64, &b, 64, w, w));
        });
        row(&format!("SAD {w}x{w}"), s, v);
        let s = best(|| {
            black_box(pixel::sse(Level::Scalar, &a, 64, &b, 64, w, w));
        });
        let v = best(|| {
            black_box(pixel::sse(simd, &a, 64, &b, 64, w, w));
        });
        row(&format!("SSE {w}x{w}"), s, v);
    }
    // Forward transform and quantisation.
    for n in 2..=5u32 {
        let n0 = 1usize << n;
        let res: Vec<i16> = (0..n0 * n0).map(|_| rng.range(-255, 255) as i16).collect();
        let mut d = vec![0i32; n0 * n0];
        let mut q = vec![0i32; n0 * n0];
        let mut f = |l: Level| {
            let k = crate::encoder::fdct::forward(l, black_box(&res), n, 0, 8, &mut d);
            crate::encoder::fdct::quantize(l, &d, k, (40, 50), 1, 1 << 14, &mut q);
        };
        let s = best(|| f(Level::Scalar));
        let v = best(|| f(simd));
        row(&format!("fdct + quantise {n0}x{n0}"), s, v);
    }
}
