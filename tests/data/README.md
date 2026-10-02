# Committed test vectors

A few of the WebM project's public VP9 test vectors, small enough to keep in
the repository so `cargo test` checks real streams without a download. Each
`name.md5` holds the MD5 of every frame the stream shows, as published next
to the stream. The full set (353 streams, about 34 MB) is fetched by
`tools/fetch-vectors.sh` into `tests/vectors/`.

Source: `https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx/<name>`
and `<name>.md5`, downloaded 2026-10-02. They are bitstream data and
expected checksums, published by the WebM project for decoder testing; no
implementation's code.

| stream | what it covers |
|---|---|
| vp90-2-02-size-08x08.webm, vp90-2-02-size-66x66.webm, vp90-2-02-size-130x132.webm, vp90-2-03-size-226x226.webm | frame sizes that are not multiples of 8 / 64 |
| vp90-2-03-deltaq.webm | delta quantisers |
| vp90-2-06-bilinear.webm | the bilinear interpolation filter |
| vp90-2-05-resize.ivf, vp90-2-13-mv-with-scaling.ivf | frame size changes, scaled references (IVF container) |
| vp90-2-10-show-existing-frame2.webm | show_existing_frame |
| vp90-2-15-fuzz-flicker.webm | a corrupt container tail |
| vp91-2-04-yuv444.webm, vp91-2-04-yuv440.webm | profile 1: 4:4:4, 4:4:0 |
| vp92-2-20-10bit-yuv420.webm | profile 2: 10-bit 4:2:0 |
| vp93-2-20-12bit-yuv422.webm | profile 3: 12-bit 4:2:2 |
