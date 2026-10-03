//! The WebM project's VP9 test vectors: every shown frame's MD5 must match.
//!
//! Run `tools/fetch-vectors.sh` first (about 34 MB into tests/vectors/).
//! Without the download this test reports that it skipped, unless
//! `VP9_REQUIRE_VECTORS=1` makes the absence a failure. `VP9_VECTOR=substr`
//! restricts the run to matching names.

mod common;

#[test]
fn test_vectors() {
    let mut paths = common::downloaded_vectors();
    if paths.is_empty() {
        assert!(
            std::env::var("VP9_REQUIRE_VECTORS").is_err(),
            "tests/vectors is empty: run tools/fetch-vectors.sh"
        );
        eprintln!("test vectors not downloaded (tools/fetch-vectors.sh); skipped");
        return;
    }
    if let Ok(f) = std::env::var("VP9_VECTOR") {
        paths.retain(|p| p.to_string_lossy().contains(&f));
    }
    let results = common::run_all(&paths);
    let passed = results.iter().filter(|o| o.passed()).count();
    for o in &results {
        if !o.passed() {
            eprintln!(
                "FAIL {}: {}/{} frames ok; {}",
                o.name,
                o.matched,
                o.expected,
                o.failure.as_deref().unwrap_or("")
            );
        }
    }
    eprintln!("{passed}/{} vectors pass", results.len());
    let key_ok = results.iter().filter(|o| o.matched > 0).count();
    eprintln!(
        "{key_ok}/{} vectors decode their first frame bit-exactly",
        results.len()
    );
    // Every vector must pass (there are no known failures).
    assert_eq!(
        passed,
        results.len(),
        "{} vectors failed",
        results.len() - passed
    );
}
