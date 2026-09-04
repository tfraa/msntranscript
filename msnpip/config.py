"""Run configuration: dataclasses built from YAML and/or CLI flags, validated once."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

from msnpip.errors import ConfigurationError

ContrastStat = Literal["beta", "t", "cohen_d"]
CorrMethod = Literal["pearson", "spearman"]
GroupLabel = str | int


@dataclass(frozen=True)
class EngineConfig:
    """Knobs forwarded to the imaging-transcriptomics engine.

    ``hemisphere`` selects what is fed to the engine; the MSN itself is always
    whole-cortex.  WARNING: ``"right"`` is a homotopic relabel of the phenotype,
    not a right-hemisphere transcriptome — AHBA expression stays left-hemisphere.
    """

    methods: tuple[Literal["pls", "corr"], ...] = ("pls",)
    atlas: str = "dk"
    hemisphere: Literal["left", "right", "both"] = "left"
    compare_hemispheres: bool = False
    regions: Literal["cort", "cort+sub"] = "cort"
    # Also run the union of the specified cases per control, e.g. {1,2,3}v0.
    pool_cases: bool = False
    n_components: int | None = 1
    var: float | None = None
    n_permutations: int = 10000
    null_method: str = "vasa"
    require_surface_null: bool = True
    # On a failed surface spin, fall back to the engine's 'auto' with a warning
    # instead of raising. The resolved null is recorded either way.
    allow_null_fallback: bool = True
    # ensemble/gsea use the spin null; ora is the template over-representation test.
    enrichment_methods: tuple[Literal["ensemble", "gsea", "ora", "none"], ...] = (
        "ensemble",
        "gsea",
        "ora",
    )
    # Resolved against the .gmt files bundled in msnpip/genes, so enrichment runs
    # offline; lake/pooled fall through to the engine's own packaged sets.
    gene_sets: tuple[str, ...] = (
        "lake",
        "pooled",
        "GO_Biological_Process_2025",
        "KEGG_2021_H",
        "DisGeNET",
    )
    # WARNING: "engine" scores every surrogate at the observed gene positions,
    # which is not a valid null for a rank-position statistic (pure-H0 FPR ~0.7).
    # Its output is written as backend "gseafrozen" and is not reportable.
    gsea_backend: Literal["corrected", "engine", "both"] = "corrected"
    # None = the engine's hardcoded 1000, regardless of n_permutations.
    gsea_engine_n_iter: int | None = None
    geneset_organism: str = "Human"
    # Category-size window, counted after intersecting each term with the ranked
    # gene universe. Off by default. Pre-specify it; tuning it on the results is
    # p-hacking.
    geneset_min_size: int = 1
    geneset_max_size: int | None = None
    # Tail is `p <= ora_p_threshold` on the UNCORRECTED spin p-value, split by the
    # sign of the ranking statistic. ORA always gets the unfiltered gene set.
    ora_p_threshold: float = 0.05
    seed: int = 1234
    n_jobs: int = 1


@dataclass(frozen=True)
class IOConfig:
    """Input source and parsing options.

    Exactly one input mode: (``freesurfer_dir`` + ``demographics``) or ``dataframe``.
    """

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
    # "distance" = 1/(1+d/n) kernel, strictly positive; "correlation" = Pearson
    # between regions' z-scored metric vectors, which allows negative edges.
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

    def validate(self) -> None:
        """Cross-field validation; raises ConfigurationError on any problem."""
        errors: list[str] = []

        has_fs = self.io.freesurfer_dir is not None and self.io.demographics is not None
        has_df = self.io.dataframe is not None
        if has_fs and has_df:
            errors.append("Set only one input mode: --input/--demographics OR --dataframe.")
        if not has_fs and not has_df:
            errors.append(
                "No input given. Provide --dataframe FILE, or --input DIR with --demographics FILE."
            )

        try:
            import imaging_transcriptomics as imt

            known = {a.id for a in imt.list_atlases()}
            if self.engine.atlas not in known:
                errors.append(f"Unknown atlas {self.engine.atlas!r}. Available: {sorted(known)}")
        except Exception as exc:  # engine import/list failure is non-fatal to validation
            errors.append(f"Could not verify atlas against the engine: {exc}")

        if "pls" in self.engine.methods and (
            (self.engine.n_components is None) == (self.engine.var is None)
        ):
            errors.append("PLS requires exactly one of engine.n_components or engine.var.")

        gcol = self.group_col or self.io.group_col
        if gcol is None and (self.case is not None or self.contrasts):
            errors.append("A contrast was requested but no group column is set.")

        if errors:
            raise ConfigurationError(
                "Invalid configuration:\n" + "\n".join(f"  • {e}" for e in errors)
            )

    def resolved_group_col(self) -> str | None:
        return self.group_col or self.io.group_col

    def to_dict(self) -> dict:
        """Plain-dict representation (Paths → str) for the YAML dump."""
        return _to_plain(asdict(self))

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
