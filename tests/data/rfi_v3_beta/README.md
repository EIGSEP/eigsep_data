# Marjum low-band RFI regression fixtures

These two compressed NPZ files are 64-integration slices from the
`marjum-2026-07` span 13 window beginning 10 hours after the visit start. The
earlier 50 ns-only spectral model over-flagged 50--70 MHz in this window. The
stationary 300 ns low-band component lowers the median absolute residual and
recovers clean cells without reducing design support.

Each file contains the air and ground autocorrelations, their complex cross,
Unix times, frequencies in MHz, integration times in seconds, switch states,
and raw `(file, row)` identities. `provenance_json` records the exact selection,
units, purpose, and SHA-256 of each source HDF5 file. Arrays were stored as
`float32`/`complex64`; the two files total about 1.4 MB.

The data are test fixtures, not released campaign products. They exist only to
lock down a failure observed in real spectra that a smooth synthetic example
does not fully represent.

`permuted_campaign_files.json` records the four complete campaign files whose
time-ordered index rows are rotated relative to raw-row order. It stores the
exact two row runs and source SHA-256 for each file. The writer regression uses
those real permutations with tiny synthetic payloads to verify that all product
arrays are restored to raw-row order without committing the full raw files.
