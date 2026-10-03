//! The test vectors committed in tests/data (see its README): every shown
//! frame must match its published MD5. Runs without any download.

mod common;

#[test]
fn committed_vectors() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "webm" || e == "ivf"))
        .collect();
    paths.sort();
    assert!(paths.len() >= 15, "tests/data is incomplete");
    let results = common::run_all(&paths);
    let failed: Vec<_> = results.iter().filter(|o| !o.passed()).collect();
    for o in &failed {
        eprintln!(
            "FAIL {}: {}/{} frames; {}",
            o.name,
            o.matched,
            o.expected,
            o.failure.as_deref().unwrap_or("")
        );
    }
    assert!(failed.is_empty());
}

#[test]
fn legacy_profile1_444_is_detected() {
    // vp91-2-04-yv444.webm predates the final profile 1 syntax: decoded by
    // the specification's rules its key frame leaves 170 000 bits of tile
    // data unread (nonzero padding) and every frame is wrong; the decoder
    // must detect it and use 4:2:0 chroma transform sizes and sub-8x8 chroma
    // motion vectors. vp91-2-04-yuv444.webm, the final syntax, must not be
    // taken for it.
    for name in ["vp91-2-04-yv444.webm", "vp91-2-04-yuv444.webm"] {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name);
        let o = common::run_vector(&p);
        assert!(o.passed(), "{name}: {o:?}");
    }
}
