"""
Design matrix, OLS fit, residualization, regional_group_contrast.
Phase 2, Tasks T2.3–T2.4.

OLS is implemented in closed form (numpy) rather than delegated to statsmodels
so the unit tests can validate it *against* statsmodels — this is where the
real-world locale/coding bugs were caught.  Categorical predictors (sex, site)
are always one-hot encoded with a dropped reference level; site coding is a
locked decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from msnpip.errors import SchemaError

logger = logging.getLogger("msnpip.stats.glm")

# Below this per-group n, contrasts/violins are flagged as small-sample (spec R7).
MIN_GROUP_N = 10


def normalize_group_value(value) -> str:
    """Canonical string for a group code so 1, 1.0 and '1' all match (issue 5)."""
    if value is None or (isinstance(value, float) and value != value):  # None / NaN
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return s


def group_mask(series: pd.Series, label) -> pd.Series:
    """Boolean mask of *series* rows whose group value matches *label*,
    robust to numeric/string differences (``1`` vs ``1.0`` vs ``'1'``)."""
    return series.map(normalize_group_value) == normalize_group_value(label)


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values).

    NaN inputs are ignored in the ranking and returned as NaN, so partially
    estimable contrast maps still get a valid correction over their finite
    entries.  Output q-values are clipped to ``[0, 1]`` and enforced monotone.
    """
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    pv = p[finite]
    m = pv.size
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0.0, 1.0)
    q[finite] = out
    return q


# ---------------------------------------------------------------------------
# T2.3 — build_design_matrix
# ---------------------------------------------------------------------------


def build_design_matrix(
    df: pd.DataFrame,
    predictors,
    *,
    add_intercept: bool = True,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Build a numeric design matrix from a list of predictor columns.

    Numeric predictors are passed through; non-numeric predictors are one-hot
    encoded (``drop_first`` drops the alphabetically-first level as the
    reference, avoiding collinearity with the intercept).

    Parameters
    ----------
    df
        Source DataFrame.
    predictors
        Iterable of column names to include.
    add_intercept
        Prepend an ``Intercept`` column of ones.
    drop_first
        Drop the first level of each categorical predictor.

    Returns
    -------
    pd.DataFrame
        Design matrix with named columns, float dtype, same row index as *df*.

    Raises
    ------
    SchemaError
        If a requested predictor is missing from *df*.
    """
    predictors = list(predictors)
    missing = [c for c in predictors if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Design predictors not found in DataFrame: {missing}. Available: {list(df.columns)}"
        )

    pieces: list[pd.DataFrame] = []
    if add_intercept:
        pieces.append(pd.DataFrame({"Intercept": np.ones(len(df))}, index=df.index))

    for col in predictors:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            pieces.append(series.astype(float).to_frame(name=col))
        else:
            dummies = pd.get_dummies(
                series.astype("category"),
                prefix=col,
                drop_first=drop_first,
                dtype=float,
            )
            pieces.append(dummies)

    design = pd.concat(pieces, axis=1)
    return design.astype(float)


# ---------------------------------------------------------------------------
# T2.3 — fit_ols
# ---------------------------------------------------------------------------


@dataclass
class OLSResult:
    """Result of an ordinary-least-squares fit."""

    params: np.ndarray  # (n_terms,)
    se: np.ndarray  # (n_terms,)
    tvalues: np.ndarray  # (n_terms,)
    pvalues: np.ndarray  # (n_terms,) two-sided
    resid: np.ndarray  # (n_obs,)
    fitted: np.ndarray  # (n_obs,)
    df_resid: int
    rank: int
    colnames: list[str]

    def index_of(self, name: str) -> int:
        try:
            return self.colnames.index(name)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Term {name!r} not in design columns {self.colnames}") from exc


def fit_ols(X, y) -> OLSResult:
    """Fit ``y = X @ beta + e`` by ordinary least squares (closed form).

    Parameters
    ----------
    X
        Design matrix — a ``pd.DataFrame`` (column names preserved) or a 2-D
        array.
    y
        Response vector.

    Returns
    -------
    OLSResult
    """
    if isinstance(X, pd.DataFrame):
        colnames = list(X.columns)
        Xmat = X.to_numpy(dtype=float)
    else:
        Xmat = np.asarray(X, dtype=float)
        if Xmat.ndim == 1:
            Xmat = Xmat[:, None]
        colnames = [f"x{i}" for i in range(Xmat.shape[1])]

    yvec = np.asarray(y, dtype=float).ravel()
    n_obs, n_terms = Xmat.shape
    if yvec.shape[0] != n_obs:
        raise ValueError(f"len(y)={yvec.shape[0]} != n_obs={n_obs}")

    beta, _, rank, _ = np.linalg.lstsq(Xmat, yvec, rcond=None)
    fitted = Xmat @ beta
    resid = yvec - fitted
    df_resid = n_obs - rank

    if df_resid <= 0:
        se = np.full(n_terms, np.nan)
        tvalues = np.full(n_terms, np.nan)
        pvalues = np.full(n_terms, np.nan)
    else:
        sigma2 = float(resid @ resid) / df_resid
        XtX = Xmat.T @ Xmat
        XtX_inv = np.linalg.pinv(XtX)
        var_beta = sigma2 * np.diag(XtX_inv)
        se = np.sqrt(np.abs(var_beta))
        with np.errstate(divide="ignore", invalid="ignore"):
            tvalues = np.where(se > 0, beta / se, np.nan)
        pvalues = 2.0 * sp_stats.t.sf(np.abs(tvalues), df_resid)

    return OLSResult(
        params=beta,
        se=se,
        tvalues=tvalues,
        pvalues=pvalues,
        resid=resid,
        fitted=fitted,
        df_resid=int(df_resid),
        rank=int(rank),
        colnames=colnames,
    )


# ---------------------------------------------------------------------------
# T2.3 — residualize
# ---------------------------------------------------------------------------


def residualize(y, covariates, *, add_intercept: bool = True, add_back_mean: bool = False):
    """Regress *y* on *covariates* and return the residuals.

    Parameters
    ----------
    y
        Response vector.
    covariates
        Design matrix of covariates (DataFrame or array).  An intercept is
        added unless one is already present and ``add_intercept=False``.
    add_intercept
        Prepend a column of ones to *covariates*.
    add_back_mean
        If True, add the grand mean of *y* back to the residuals so the
        residualized values retain the original scale/location.

    Returns
    -------
    np.ndarray
        Residuals (1-D), same length as *y*.
    """
    yvec = np.asarray(y, dtype=float).ravel()

    if isinstance(covariates, pd.DataFrame):
        C = covariates.to_numpy(dtype=float)
    else:
        C = np.asarray(covariates, dtype=float)
        if C.ndim == 1:
            C = C[:, None]

    if add_intercept:
        C = np.column_stack([np.ones(len(yvec)), C])

    beta, _, _, _ = np.linalg.lstsq(C, yvec, rcond=None)
    resid = yvec - C @ beta
    if add_back_mean:
        resid = resid + yvec.mean()
    return resid


# ---------------------------------------------------------------------------
# T2.4 — regional_group_contrast
# ---------------------------------------------------------------------------


@dataclass
class GroupContrastResult:
    """Per-region case-vs-control contrast map.

    ``region_labels`` use the ``"{hemi}_{aparc_label}"`` format so the map feeds
    :func:`msnpip.atlas_align.align_strength_to_atlas` directly.
    """

    regional_stat: np.ndarray  # (n_regions,)
    region_labels: list[str]
    stat_type: str  # "beta" | "t" | "cohen_d"
    covariates: list[str] = field(default_factory=list)
    n_case: int = 0
    n_control: int = 0
    group_term: str = ""  # design column the statistic was read from
    atlas: str = "dk"
    hemisphere: str = "left"
    regions: str = "cort"
    # Full per-region group-effect statistics (always populated), so the report
    # can highlight significant regions with beta + t + p + FDR regardless of
    # which ``stat_type`` was selected for the exported contrast map.
    beta: np.ndarray | None = None
    tvalue: np.ndarray | None = None
    pvalue: np.ndarray | None = None
    pvalue_fdr: np.ndarray | None = None
    cohen_d: np.ndarray | None = None

    def stats_table(self) -> pd.DataFrame:
        """Per-region group-effect table (region, beta, t, cohen_d, p, fdr)."""
        n = len(self.region_labels)
        nan = np.full(n, np.nan)
        return pd.DataFrame(
            {
                "region": list(self.region_labels),
                "beta": self.beta if self.beta is not None else nan,
                "t": self.tvalue if self.tvalue is not None else nan,
                "cohen_d": self.cohen_d if self.cohen_d is not None else nan,
                "p": self.pvalue if self.pvalue is not None else nan,
                "fdr": self.pvalue_fdr if self.pvalue_fdr is not None else nan,
            }
        )

    def significant_table(self, alpha: float = 0.05) -> pd.DataFrame:
        """FDR-significant regions (``fdr < alpha``), sorted by ascending FDR."""
        tbl = self.stats_table()
        sig = tbl[tbl["fdr"] < alpha].copy()
        return sig.sort_values(["fdr", "p"], kind="mergesort").reset_index(drop=True)


def _cohen_d(case_vals: np.ndarray, control_vals: np.ndarray) -> float:
    """Cohen's d with pooled SD (ddof=1)."""
    nc, nk = len(case_vals), len(control_vals)
    if nc < 2 or nk < 2:
        return np.nan
    vc = case_vals.var(ddof=1)
    vk = control_vals.var(ddof=1)
    pooled = np.sqrt(((nc - 1) * vc + (nk - 1) * vk) / (nc + nk - 2))
    if pooled == 0:
        return np.nan
    return float((case_vals.mean() - control_vals.mean()) / pooled)


def regional_group_contrast(
    strength_maps,
    df: pd.DataFrame,
    schema,
    *,
    case_label,
    control_label,
    group_col: str | None = None,
    covariates=(),
    stat: str = "beta",
) -> GroupContrastResult:
    """Contrast node strength between two groups, per region.

    For each region, fits ``strength ~ group + covariates`` by OLS (group coded
    1 = case, 0 = control) and exports a per-region statistic:

    - ``"beta"``  (default): the group coefficient.
    - ``"t"``     : its t-statistic.
    - ``"cohen_d"``: standardized mean difference on covariate-residualized
      strength (Cohen's d, pooled SD).

    Parameters
    ----------
    strength_maps
        :class:`msnpip.msn.construct.StrengthMaps`.
    df
        DataFrame with the group column and covariates.  Rows are aligned to
        ``strength_maps.subject_ids`` by ``schema.id_col``.
    schema
        Column schema (provides ``id_col`` and the default ``group_col``).
    case_label, control_label
        Values in the group column identifying each arm.
    group_col
        Group column name; defaults to ``schema.group_col``.
    covariates
        Covariate column names (numeric pass-through, categorical one-hot).
    stat
        Which statistic to export.

    Returns
    -------
    GroupContrastResult

    Raises
    ------
    SchemaError
        If the group column is missing, or a group arm is empty.
    """
    if stat not in ("beta", "t", "cohen_d"):
        raise ValueError(f"stat must be 'beta'/'t'/'cohen_d', got {stat!r}")

    group_col = group_col or getattr(schema, "group_col", None)
    if group_col is None or group_col not in df.columns:
        raise SchemaError(
            f"Group column {group_col!r} not found. Set schema.group_col or pass group_col."
        )

    covariates = list(covariates)

    # Align df rows to the strength_maps subject order.
    id_col = schema.id_col
    df_idx = df.set_index(df[id_col].astype(str))
    try:
        aligned = df_idx.loc[strength_maps.subject_ids]
    except KeyError as exc:
        raise SchemaError(
            "Some strength-map subject IDs are absent from the DataFrame — "
            "cannot align for the contrast."
        ) from exc

    group_series = aligned[group_col]
    is_case = group_mask(group_series, case_label).to_numpy()
    is_control = group_mask(group_series, control_label).to_numpy()
    keep = is_case | is_control
    if keep.sum() == 0:
        raise SchemaError(
            f"No subjects matched case_label={case_label!r} / "
            f"control_label={control_label!r} in column {group_col!r}."
        )

    sub = aligned.loc[keep].copy()
    strength = strength_maps.strength[keep]
    group_indicator = group_mask(sub[group_col], case_label).astype(float)
    n_case = int(group_indicator.sum())
    n_control = int(len(sub) - n_case)
    if n_case < 1 or n_control < 1:
        raise SchemaError(f"Each arm needs ≥1 subject (case={n_case}, control={n_control}).")
    if n_case < MIN_GROUP_N or n_control < MIN_GROUP_N:
        logger.warning(
            "Small group(s): case=%d, control=%d (< %d). Per-region contrasts and the "
            "spatial null may be unstable; interpret with caution and prefer cohen_d.",
            n_case,
            n_control,
            MIN_GROUP_N,
        )

    # Build the design once: intercept + group + covariates.
    design_input = sub[covariates].copy() if covariates else pd.DataFrame(index=sub.index)
    design_input.insert(0, "group", group_indicator.to_numpy())
    design = build_design_matrix(design_input, list(design_input.columns), add_intercept=True)
    group_term = "group"

    # Design-rank guardrail: a rank-deficient or near-saturated design (too many
    # covariate terms for the sample) makes per-region t/p NaN while beta is still
    # emitted via pinv. Warn once, up front, so this isn't silently buried in a
    # column of NaNs across every region.
    n_obs, n_terms = design.shape
    rank = int(np.linalg.matrix_rank(design.to_numpy(dtype=float)))
    if rank < n_terms or (n_obs - rank) <= 1:
        logger.warning(
            "Design is rank-deficient / near-saturated: n_obs=%d, terms=%d, rank=%d "
            "(residual df=%d). Per-region t/p will be NaN (beta still reported); "
            "reduce covariates or add subjects.",
            n_obs,
            n_terms,
            rank,
            n_obs - rank,
        )

    n_regions = strength.shape[1]
    regional_stat = np.full(n_regions, np.nan)

    # Always fit the OLS group model per region to collect beta + t + p; this
    # backs the report's significant-region highlights and the FDR correction,
    # independent of which statistic is exported as the contrast map.
    beta_arr = np.full(n_regions, np.nan)
    t_arr = np.full(n_regions, np.nan)
    p_arr = np.full(n_regions, np.nan)
    gi = design.columns.get_loc(group_term)
    for r in range(n_regions):
        res = fit_ols(design, strength[:, r])
        beta_arr[r] = res.params[gi]
        t_arr[r] = res.tvalues[gi]
        p_arr[r] = res.pvalues[gi]
    fdr_arr = benjamini_hochberg(p_arr)

    # Always compute Cohen's d (covariate-residualised standardized mean diff) so
    # the per-region stats table can report it alongside beta/t/p/FDR.
    cov_design = (
        build_design_matrix(sub[covariates], covariates, add_intercept=True) if covariates else None
    )
    case_mask = group_indicator.to_numpy().astype(bool)
    d_arr = np.full(n_regions, np.nan)
    for r in range(n_regions):
        y = strength[:, r]
        if cov_design is not None:
            y = residualize(y, cov_design, add_intercept=False)
        d_arr[r] = _cohen_d(y[case_mask], y[~case_mask])

    if stat == "cohen_d":
        regional_stat = d_arr.copy()
    elif stat == "beta":
        regional_stat = beta_arr.copy()
    else:
        regional_stat = t_arr.copy()

    logger.info(
        "regional_group_contrast: stat=%s case=%d control=%d covariates=%s → %d regions",
        stat,
        n_case,
        n_control,
        covariates,
        n_regions,
    )

    return GroupContrastResult(
        regional_stat=regional_stat,
        region_labels=list(strength_maps.region_labels),
        stat_type=stat,
        covariates=covariates,
        n_case=n_case,
        n_control=n_control,
        group_term=group_term,
        atlas=strength_maps.atlas,
        hemisphere=strength_maps.hemisphere,
        regions=strength_maps.regions,
        beta=beta_arr,
        tvalue=t_arr,
        pvalue=p_arr,
        pvalue_fdr=fdr_arr,
        cohen_d=d_arr,
    )
