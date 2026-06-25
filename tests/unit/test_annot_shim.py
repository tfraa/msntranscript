"""Correctness test for the .annot→GIFTI spin-null shim (issue 6 / C3).

The DK spin null only works if the shim converts the FreeSurfer .annot into a
GIFTI label image whose label table lets neuromaps drop the medial-wall/unknown
parcels, leaving exactly the 34 cortical DK regions per hemisphere.
"""

from __future__ import annotations

import imaging_transcriptomics as imt
import neuromaps.nulls.spins as spins

from msnpip.engine import enable_annot_surface_nulls


def test_shim_yields_34_cortical_parcels_for_dk():
    enable_annot_surface_nulls()
    atlas = imt.get_atlas("dk")
    annot_lh = str(atlas.surface_paths[0])

    gii = spins.load_gifti(annot_lh)
    labels = gii.agg_data()
    labeltable = gii.labeltable.get_labels_as_dict()

    # fsaverage5 has 10242 vertices per hemisphere.
    assert labels.shape[0] == 10242
    # Parcels remaining after neuromaps drops PARCIGNORE names == 34 DK cortical regions.
    kept = {lab for lab in set(labels.tolist()) if labeltable.get(lab) not in spins.PARCIGNORE}
    assert len(kept) == 34
