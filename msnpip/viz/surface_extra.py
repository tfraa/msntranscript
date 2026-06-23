"""
plot_surface_with_dorsal — extends engine plotting with a dorsal (top) view.
Reuses imaging_transcriptomics.outputs.brain primitives.
Phase 4, Task T4.3.

The engine ships lateral + medial cortical views but not a dorsal (top-down)
view.  When only lateral/medial are requested we delegate straight to the
engine's ``plot_cortical_surface_map``.  When ``dorsal`` is requested we mirror
the engine's matplotlib layout (RdBu_r, symmetric norm, horizontal colorbar)
and add a top-down panel rendered with ``elev≈90``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from imaging_transcriptomics import plotting
from imaging_transcriptomics.outputs import brain

from msnpip.viz.theme import configure_theme

logger = logging.getLogger("msnpip.viz.surface_extra")

# Azimuths matching the engine's per-hemisphere lateral/medial panels.
_AZIM_LATERAL = {"left": 180.0, "right": 0.0}
_AZIM_MEDIAL = {"left": 0.0, "right": 180.0}
_AZIM_DORSAL = 270.0
_ELEV_DORSAL = 90.0


def plot_surface_with_dorsal(
    table,
    *,
    atlas_id: str,
    value_column: str,
    title: str,
    output_path,
    views: tuple[str, ...] = ("lateral", "medial", "dorsal"),
) -> Path | None:
    """Render a cortical map across the requested views, including dorsal.

    Parameters
    ----------
    table
        Region table (``id, label, hemisphere, structure, <value_column>``) —
        the output of :func:`msnpip.atlas_align.to_region_table`.
    atlas_id, value_column, title, output_path
        Forwarded to the engine renderer.
    views
        Any of ``"lateral"``, ``"medial"``, ``"dorsal"`` in display order.

    Returns
    -------
    Path | None
        The written PNG path, or ``None`` if surface assets are unavailable
        (no surface paths / empty frames / no finite values).
    """
    configure_theme()
    views = tuple(views)
    unknown = [v for v in views if v not in ("lateral", "medial", "dorsal")]
    if unknown:
        raise ValueError(f"Unknown view(s) {unknown}; expected lateral/medial/dorsal.")

    # Lateral/medial only → the engine already does this well.
    if "dorsal" not in views:
        return plotting.plot_cortical_surface_map(
            table,
            atlas_id=atlas_id,
            value_column=value_column,
            title=title,
            output_path=output_path,
        )

    _, plt = brain.matplotlib_backend()

    hemi_frames = brain.surface_value_frames(table)
    if not hemi_frames:
        logger.warning("plot_surface_with_dorsal: no hemisphere frames — skipping.")
        return None
    atlas = brain.get_atlas(atlas_id)
    if getattr(atlas, "surface_paths", None) is None:
        logger.warning("plot_surface_with_dorsal: atlas %s has no surface paths.", atlas_id)
        return None
    inflated_paths = brain.surface_mesh_paths(atlas_id)
    if inflated_paths is None:
        return None

    finite = table[value_column].to_numpy(dtype=float, copy=False)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    vmax = float(np.max(np.abs(finite))) or 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    present = [h for h in ("left", "right") if h in hemi_frames]
    hemi_index = {"left": 0, "right": 1}

    def _render_hemi(ax, hemi: str, azim: float, elev: float | None) -> None:
        idx = hemi_index[hemi]
        coords, triangles = brain.load_surface_mesh(inflated_paths[idx])
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

    # Build the ordered panel list.
    panels: list[tuple[str, object]] = []
    for view in views:
        if view == "lateral":
            for h in present:
                panels.append(
                    (f"{h} lateral", lambda ax, h=h: _render_hemi(ax, h, _AZIM_LATERAL[h], None))
                )
        elif view == "medial":
            for h in present:
                panels.append(
                    (f"{h} medial", lambda ax, h=h: _render_hemi(ax, h, _AZIM_MEDIAL[h], None))
                )
        else:  # dorsal — one panel, all present hemispheres from the top

            def _render_dorsal(ax):
                for h in present:
                    _render_hemi(ax, h, _AZIM_DORSAL, _ELEV_DORSAL)

            panels.append(("dorsal", _render_dorsal))

    n = len(panels)
    fig = plt.figure(figsize=(3.3 * n, 3.8))
    fig.patch.set_facecolor("white")

    margin = 0.02
    panel_w = (1.0 - (n + 1) * margin) / n
    for i, (label, render) in enumerate(panels):
        left = margin + i * (panel_w + margin)
        ax = fig.add_axes([left, 0.20, panel_w, 0.70], projection="3d")
        render(ax)
        fig.text(
            left + panel_w / 2.0,
            0.90,
            label,
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="#334155",
        )

    fig.text(0.02, 0.97, title, ha="left", va="top", fontweight="bold", fontsize=12.0)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar_ax = fig.add_axes([0.36, 0.08, 0.28, 0.035])
    colorbar = fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    colorbar.set_ticks([-vmax, 0.0, vmax])
    colorbar.set_label(value_column, fontsize=9.5)

    out = brain.save_figure(fig, Path(output_path))
    logger.info("plot_surface_with_dorsal: wrote %s (views=%s)", out, views)
    return out
