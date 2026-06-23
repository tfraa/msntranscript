# Adding a new atlas to msnpip

> Stub — implement after Phase 1 (T1.6) is complete.

New atlases require:
1. Engine support (verify with `imt.list_atlases()`)
2. FreeSurfer re-parcellation to produce per-subject aparc-equivalent stats
3. A feature-extraction hook registered in `io/readers.py`
4. An alignment test in `tests/unit/test_atlas_align.py`

For Schaefer/Glasser, the main bottleneck is step 2 (FreeSurfer
re-parcellation), not the msnpip code.
