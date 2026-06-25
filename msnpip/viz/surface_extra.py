"""
plot_surface_with_dorsal — cortical surface maps with lateral/medial/dorsal views.
Reuses imaging_transcriptomics.outputs.brain primitives.
Phase 4, Task T4.3.

Renders a regional map on the cortical surface for the hemispheres present in the
table (both, when available), across lateral, medial and a dorsal (top-down,
``elev≈90``) view.  The mesh family is selectable (``pial`` or ``inflated``).
A title and subtitle make explicit what the map is and where it comes from.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from imaging_transcriptomics.outputs import brain

from msnpip.viz.theme import configure_theme

logger = logging.getLogger("msnpip.viz.surface_extra")

# Azimuths matching the engine's per-hemisphere lateral/medial panels.
_AZIM_LATERAL = {"left": 180.0, "right": 0.0}
_AZIM_MEDIAL = {"left": 0.0, "right": 180.0}
_AZIM_DORSAL = 270.0
_ELEV_DORSAL = 90.0
_HEMI_LABEL = {"left": "L", "right": "R"}


def plot_surface_with_dorsal(
    table,
    *,
    atlas_id: str,
    value_column: str,
    title: str,
    output_path,
    views: tuple[str, ...] = ("lateral", "medial", "dorsal"),
    mesh_kind: str = "pial",
    subtitle: str | None = None,
    diverging: bool = True,
    cmap_name: str | None = None,
) -> Path | None:
    """Render a cortical map across the requested views and hemispheres.

    Parameters
    ----------
    table
        Region table (``id, label, hemisphere, structure, <value_column>``) —
        the output of :func:`msnpip.atlas_align.to_region_table`.  Both
        hemispheres are rendered if present.
    atlas_id, value_column
        Atlas id and the value column to colour by.
    title
        Main figure title (what the map is).
    output_path
        PNG path to write.
    views
        Any of ``"lateral"``, ``"medial"``, ``"dorsal"`` in display order.
    mesh_kind
        ``"pial"`` (anatomical) or ``"inflated"`` (smoothed) surface.
    subtitle
        Optional second line (provenance: atlas, surface, measure).
    diverging
        ``True`` (default): symmetric diverging map centred at 0 (``RdBu_r``),
        for signed contrast statistics.  ``False``: sequential map spanning the
        data range (``viridis``), for non-negative quantities like node
        strength.  Non-significant regions passed as NaN render at the diverging
        centre colour (neutral white), which is how the significant-only map is
        drawn.
    cmap_name
        Override the colormap name; defaults to ``RdBu_r`` (diverging) or
        ``viridis`` (sequential).

    Returns
    -------
    Path | None
        The written PNG path, or ``None`` if surface assets are unavailable.
    """
    configure_theme()
    views = tuple(views)
    unknown = [v for v in views if v not in ("lateral", "medial", "dorsal")]
    if unknown:
        raise ValueError(f"Unknown view(s) {unknown}; expected lateral/medial/dorsal.")
    if mesh_kind not in ("pial", "inflated"):
        raise ValueError(f"mesh_kind must be 'pial'/'inflated', got {mesh_kind!r}")

    _, plt = brain.matplotlib_backend()

    hemi_frames = brain.surface_value_frames(table)
    if not hemi_frames:
        logger.warning("plot_surface_with_dorsal: no hemisphere frames — skipping.")
        return None
    atlas = brain.get_atlas(atlas_id)
    if getattr(atlas, "surface_paths", None) is None:
        logger.warning("plot_surface_with_dorsal: atlas %s has no surface paths.", atlas_id)
        return None
    mesh_paths = brain.surface_mesh_paths(atlas_id, mesh_kind=mesh_kind)
    if mesh_paths is None:
        return None

    finite = table[value_column].to_numpy(dtype=float, copy=False)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    if diverging:
        vmax = float(np.max(np.abs(finite))) or 1.0
        vmin = -vmax
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cbar_ticks = [vmin, 0.0, vmax]
        cmap = plt.get_cmap(cmap_name or "RdBu_r")
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cbar_ticks = [vmin, (vmin + vmax) / 2.0, vmax]
        cmap = plt.get_cmap(cmap_name or "viridis")
    present = [h for h in ("left", "right") if h in hemi_frames]
    hemi_index = {"left": 0, "right": 1}

    def _render_hemi(ax, hemi: str, azim: float, elev: float | None) -> None:
        idx = hemi_index[hemi]
        coords, triangles = brain.load_surface_mesh(mesh_paths[idx])
        label_array, code_to_name = brain.load_surface_parcellation(str(atlas.surface_paths[idx]))
        vertex_values = brain.vertex_values_for_hemisphere(
            hemi_frames[hemi],
            value_column=value_column,
            label_array=label_array,
            code_to_name=code_to_name,
        )
        kwargs = dict(azim=azim, cmap=cmap, norm=norm)
        if elev is not None:
            kwargs["elev"] = elev
        brain.surface_view(ax, coords, triangles, vertex_values, **kwargs)

    # Build the ordered panel list (per-hemisphere lateral/medial + a combined dorsal).
    panels: list[tuple[str, object]] = []
    for view in views:
        if view == "lateral":
            for h in present:
                panels.append(
                    (
                        f"{_HEMI_LABEL[h]} lateral",
                        lambda ax, h=h: _render_hemi(ax, h, _AZIM_LATERAL[h], None),
                    )
                )
        elif view == "medial":
            for h in present:
                panels.append(
                    (
                        f"{_HEMI_LABEL[h]} medial",
                        lambda ax, h=h: _render_hemi(ax, h, _AZIM_MEDIAL[h], None),
                    )
                )
        else:  # dorsal — one panel, all present hemispheres from the top

            def _render_dorsal(ax):
                for h in present:
                    _render_hemi(ax, h, _AZIM_DORSAL, _ELEV_DORSAL)

            panels.append(("dorsal", _render_dorsal))

    n = len(panels)
    fig = plt.figure(figsize=(3.0 * n, 3.9))
    fig.patch.set_facecolor("white")

    margin = 0.02
    panel_w = (1.0 - (n + 1) * margin) / n
    for i, (label, render) in enumerate(panels):
        left = margin + i * (panel_w + margin)
        ax = fig.add_axes([left, 0.18, panel_w, 0.66], projection="3d")
        render(ax)
        fig.text(
            left + panel_w / 2.0,
            0.86,
            label,
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="#334155",
        )

    fig.text(0.02, 0.975, title, ha="left", va="top", fontweight="bold", fontsize=12.5)
    sub = subtitle or f"{atlas_id} atlas · {mesh_kind} surface · {value_column}"
    fig.text(0.02, 0.93, sub, ha="left", va="top", fontsize=9.0, color="#555555")

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar_ax = fig.add_axes([0.37, 0.07, 0.26, 0.032])
    colorbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    colorbar.set_ticks(cbar_ticks)
    colorbar.set_label(value_column, fontsize=9.5)

    out = brain.save_figure(fig, Path(output_path))
    logger.info(
        "plot_surface_with_dorsal: wrote %s (views=%s, mesh=%s, hemis=%s)",
        out,
        views,
        mesh_kind,
        present,
    )
    return out
