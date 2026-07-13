"""
PipelineConfig and sub-configs (dataclasses).
Phase 5, Task T5.1 (EngineConfig was added early in Phase 3).

See msnpip_refactor_spec.md §4.1.  The config is the single source of truth for
a run: built from a YAML file and/or CLI flags, validated once, then serialized
into the output manifest for provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

from msnpip.errors import ConfigurationError

# --- type aliases (spec §4.1) ------------------------------------------------
StrengthAgg = Literal["mean", "sum"]
ContrastStat = Literal["beta", "t", "cohen_d"]
CorrMethod = Literal["pearson", "spearman"]
GroupLabel = str | int


@dataclass(frozen=True)
class EngineConfig:
    """Knobs forwarded to the imaging-transcriptomics engine.

    The null model is FIXED to ``vasa`` (surface spin) and is not CLI-exposed;
    ``require_surface_null`` makes the wrapper hard-fail rather than let a silent
    grouped-shuffle fallback reach a figure.  ``hemisphere`` here selects what is
    fed to the *engine* — the MSN itself is always whole-cortex (both).
    """

    methods: tuple[Literal["pls"], ...] = ("pls",)
    atlas: str = "dk"
    hemisphere: Literal["left", "both"] = "left"
    compare_hemispheres: bool = False
    regions: Literal["cort", "cort+sub"] = "cort"
    # Run a supplementary pooled contrast (union of the specified cases per
    # control, e.g. 1v0/2v0/3v0 → {1,2,3}v0) alongside the per-contrast analyses.
    pool_cases: bool = False
    n_components: int | None = 1
    var: float | None = None
    n_permutations: int = 10000
    null_method: str = "vasa"
    require_surface_null: bool = True
    # If the surface spin still fails, fall back to the engine's 'auto'
    # (vasa → alexander_bloch → random) instead of hard-failing. Records the
    # resolved null; a degraded (random) null is warned, not fatal.
    allow_null_fallback: bool = True
    # ensemble (GCEA) + gsea run on the spin null (rigorous); ora is the template
    # over-representation test (candidate mechanisms, weight-ranked tails).
    enrichment_methods: tuple[Literal["ensemble", "gsea", "ora", "none"], ...] = (
        "ensemble",
        "gsea",
        "ora",
    )
    # Names resolve to the .gmt files bundled in ``msnpip/genes`` so enrichment
    # runs fully offline (lake/pooled fall back to the engine's packaged sets).
    gene_sets: tuple[str, ...] = (
        "lake",
        "pooled",
        "GO_Biological_Process_2025",
        "KEGG_2021_H",
        "DisGeNET",
    )
    geneset_organism: str = "Human"
    ora_p_threshold: float | None = None
    # Weight-ranking cut (|standardized loading|) defining the PLS1± tails for the
    # template ORA. 3.0 = the classic Z>3 cut used by the source literature.
    ora_z_cut: float = 3.0
    seed: int = 1234
    n_jobs: int = 1


@dataclass(frozen=True)
class IOConfig:
    """Input source + parsing options.  Exactly one input mode must be set:
    (``freesurfer_dir`` + ``demographics``) OR ``dataframe``."""

    freesurfer_dir: Path | None = None
    demographics: Path | None = None
    dataframe: Path | None = None
    sep: str | None = None
    decimal: str | None = None
    sheet: str | int | None = 0
    id_col: str | None = None
    group_col: str | None = None
    min_id_match_rate: float = 0.95


@dataclass(frozen=True)
class MSNConfig:
    features: tuple[str, ...] = ("SurfArea", "GrayVol", "ThickAvg", "MeanCurv", "GausCurv")
    strength_agg: StrengthAgg = "sum"  # node strength = sum of edge weights (Tomasella et al.)
    # Edge definition: "distance" = 1/(1+d/n) kernel (default, strictly positive);
    # "correlation" = Pearson between regions' z-scored metric vectors (canonical
    # morphometric similarity, Seidlitz 2018 / Morgan 2019; allows negative edges).
    similarity: Literal["distance", "correlation"] = "distance"


@dataclass(frozen=True)
class GLMConfig:
    predictors: tuple[str, ...] = ()
    one_hot_always: tuple[str, ...] = ("site", "scanner")
    reference_levels: dict[str, str] = field(default_factory=dict)
    contrast_stat: ContrastStat = "beta"


@dataclass(frozen=True)
class CorrelationConfig:
    variables: tuple[str, ...] = ()
    method: CorrMethod = "spearman"
    scope: Literal["global", "regional"] = "global"
    within_group: GroupLabel | None = None


@dataclass(frozen=True)
class PipelineConfig:
    io: IOConfig
    output: Path
    group_col: str | None = None
    case: GroupLabel | None = None
    control: GroupLabel | None = None
    contrasts: tuple[tuple[GroupLabel, GroupLabel], ...] | None = None
    msn: MSNConfig = field(default_factory=MSNConfig)
    glm: GLMConfig = field(default_factory=GLMConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    save_all: bool = True
    save_figures: bool = True
    verbose: bool = False

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Cross-field validation; raises ConfigurationError on any problem."""
        errors: list[str] = []

        # Exactly one input mode.
        has_fs = self.io.freesurfer_dir is not None and self.io.demographics is not None
        has_df = self.io.dataframe is not None
        if has_fs and has_df:
            errors.append("Set only one input mode: --input/--demographics OR --dataframe.")
        if not has_fs and not has_df:
            errors.append(
                "No input given. Provide --dataframe FILE, or --input DIR with --demographics FILE."
            )

        # Atlas must be known to the engine (validated live).
        try:
            import imaging_transcriptomics as imt

            known = {a.id for a in imt.list_atlases()}
            if self.engine.atlas not in known:
                errors.append(f"Unknown atlas {self.engine.atlas!r}. Available: {sorted(known)}")
        except Exception as exc:  # engine import/list failure is non-fatal to validation
            errors.append(f"Could not verify atlas against the engine: {exc}")

        # PLS needs exactly one of n_components / var.
        if "pls" in self.engine.methods and (
            (self.engine.n_components is None) == (self.engine.var is None)
        ):
            errors.append("PLS requires exactly one of engine.n_components or engine.var.")

        # Need a group column to define a contrast.
        gcol = self.group_col or self.io.group_col
        if gcol is None and (self.case is not None or self.contrasts):
            errors.append("A contrast was requested but no group column is set.")

        if errors:
            raise ConfigurationError(
                "Invalid configuration:\n" + "\n".join(f"  • {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    def resolved_group_col(self) -> str | None:
        return self.group_col or self.io.group_col

    def to_dict(self) -> dict:
        """Plain-dict representation (Paths → str) for the manifest / YAML dump."""
        return _to_plain(asdict(self))

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        data = dict(data)
        io = IOConfig(
            **_coerce_paths(data.pop("io", {}), ("freesurfer_dir", "demographics", "dataframe"))
        )
        msn = MSNConfig(**_tuplify(data.pop("msn", {}), ("features",)))
        glm = GLMConfig(**_tuplify(data.pop("glm", {}), ("predictors", "one_hot_always")))
        corr = CorrelationConfig(**_tuplify(data.pop("correlation", {}), ("variables",)))
        engine = EngineConfig(
            **_tuplify(data.pop("engine", {}), ("methods", "enrichment_methods", "gene_sets"))
        )
        output = Path(data.pop("output"))
        contrasts = data.pop("contrasts", None)
        if contrasts is not None:
            contrasts = tuple(tuple(pair) for pair in contrasts)
        return cls(
            io=io,
            output=output,
            msn=msn,
            glm=glm,
            correlation=corr,
            engine=engine,
            contrasts=contrasts,
            **data,
        )

    def merged_with(self, **overrides) -> PipelineConfig:
        """Return a copy with top-level fields overridden (CLI > YAML)."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_plain(obj):
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _coerce_paths(d: dict, keys) -> dict:
    d = dict(d)
    for k in keys:
        if d.get(k) is not None:
            d[k] = Path(d[k])
    return d


def _tuplify(d: dict, keys) -> dict:
    d = dict(d)
    for k in keys:
        if d.get(k) is not None and not isinstance(d[k], tuple):
            d[k] = tuple(d[k])
    return d
