"""
PipelineConfig and sub-configs (dataclasses).

Most of the config tree lands in Phase 5 (Task T5.1); see msnpip_refactor_spec.md
§4.1 for the full set.  ``EngineConfig`` is defined here early because the Phase 3
engine wrapper (``engine.py``) imports it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EngineConfig:
    """Knobs forwarded to the imaging-transcriptomics engine.

    The null model is FIXED to ``vasa`` (surface spin) and is not CLI-exposed;
    ``require_surface_null`` makes the wrapper hard-fail rather than let a silent
    grouped-shuffle fallback reach a figure.
    """

    methods: tuple[Literal["pls", "corr"], ...] = ("pls", "corr")
    atlas: str = "dk"
    hemisphere: Literal["left", "both"] = "left"
    compare_hemispheres: bool = False
    regions: Literal["cort", "cort+sub"] = "cort"
    n_components: int | None = 1
    var: float | None = None
    n_permutations: int = 10000
    null_method: str = "vasa"
    require_surface_null: bool = True
    enrichment_methods: tuple[Literal["ensemble", "gsea", "ora", "none"], ...] = ("ensemble", "gsea")
    gene_sets: tuple[str, ...] = (
        "lake",
        "pooled",
        "GO_Biological_Process_2025",
        "KEGG_2021_Human",
        "DisGeNET",
    )
    geneset_organism: str = "Human"
    ora_p_threshold: float | None = None
    seed: int = 1234
    n_jobs: int = 1
