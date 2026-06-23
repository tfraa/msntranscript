# Adding a new atlas to msnpip

msnpip delegates all atlas knowledge to the imaging-transcriptomics engine, so adding an
atlas is mostly an *engine* and *data* task — the msnpip code is atlas-agnostic. Switching
`--atlas dk → --atlas <other>` requires **no code change**, provided the four prerequisites
below are met.

## Prerequisites

1. **Engine support.** The atlas must be registered in the engine:

   ```python
   import imaging_transcriptomics as imt
   print([a.id for a in imt.list_atlases()])   # your atlas id must appear here
   ```

   `msnpip list-atlases` proxies this. `PipelineConfig.validate()` checks the atlas live and
   fails fast if it is unknown.

2. **Morphometric features in the same naming convention.** msnpip expects feature columns
   named `{hemisphere}_{region}_{metric}` where `{region}` matches the engine's label names
   for that atlas. For non-DK parcellations this means re-running FreeSurfer (or your
   morphometry tool) with the target parcellation so the per-subject stats use the new region
   names. **This is the real bottleneck** — not msnpip.

3. **Alignment is automatic.** `atlas_align.align_strength_to_atlas` reorders/subsets the MSN
   regional map to the engine's canonical `(hemisphere, label)` order and **raises**
   `AtlasAlignmentError` on any unmatched region — it never silently zero-fills. If the new
   atlas's labels differ from your feature columns, you will get a clear error listing the
   missing regions.

4. **A small alignment test.** Add a case to `tests/unit/test_atlas_align.py` asserting the
   exact label order for the new atlas (mirror the DK test), so future engine updates can't
   silently reorder regions.

## What you do NOT need to change

- No new module, subclass, or registry entry — there is no atlas ABC in v2.
- No change to MSN construction, stats, the engine wrapper, or the report.
- Gene count / FDR denominators come from the engine per atlas; record the exact value the
  engine reports for your atlas in your manuscript.

## Region scope

`--regions cort` uses cortex only (the clean match for `aparc.stats` input). `cort+sub` adds
the engine's packaged subcortical regions; only use it if your features include subcortical
values, and note (per the methods doc) that subcortical rows use grouped-shuffle nulls while
cortex uses the surface spin.
