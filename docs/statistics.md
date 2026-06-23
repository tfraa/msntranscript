# Statistical methods

> Stub — expand from `msnpip_methods_theory.md` after Phase 3 (engine
> integration) and Phase 5 (report builder) are complete.

Key references and decisions are documented in
`msnpip_methods_theory.md` (theory companion) and
`msnpip_refactor_spec.md` §0.0 (locked parameter table).

Short summary of the two null layers:

- **Layer A (spatial)** — handled by the engine via the vasa surface spin.
  Covers the "is this gene/pathway association stronger than under spatially-
  autocorrelated random brain maps?" question.
- **Layer B (sampling)** — handled by msnpip via subject-level bootstrap
  (optional, not built in v2).  Covers "is the contrast map itself stable
  under subject resampling?"
