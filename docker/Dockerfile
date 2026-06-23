# msnpip 2.0 — runtime image (spec §9)
# Matplotlib (Agg) rendering path only; BrainSpace is NOT installed, so no GL/xvfb needed.
FROM python:3.12-slim

ENV MPLBACKEND=Agg \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git + build tools: the engine dependency resolves from a pinned git commit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY msnpip ./msnpip

RUN python -m pip install --upgrade pip \
    && python -m pip install .

# Pre-warm the neuromaps fsaverage surface cache so cortical plots and the
# vasa surface null work offline at run time (spec §9).
RUN python -c "import imaging_transcriptomics.outputs.brain as b; \
    getattr(b, 'fetch_standard_surface_meshes', lambda: None)()" || true

# Non-root user; mounted volumes for data (ro) and output (rw).
RUN useradd --create-home --uid 1000 msnpip
USER msnpip
VOLUME ["/data", "/out"]
WORKDIR /out

ENTRYPOINT ["msnpip"]
CMD ["--help"]
