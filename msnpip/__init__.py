"""msnpip 2.0 — Morphometric Similarity Network Imaging Transcriptomics Pipeline.

Public surface::

    from msnpip import Pipeline, run_pipeline, PipelineConfig
"""

from __future__ import annotations

from msnpip.config import PipelineConfig
from msnpip.pipeline import Pipeline, run_pipeline

__version__ = "2.0.0"
__all__ = ["Pipeline", "PipelineConfig", "__version__", "run_pipeline"]
