"""
pytest configuration and shared session-scoped fixtures.
"""
from __future__ import annotations

import pytest

from tests.fixtures.synthetic import make_synthetic_cohort


@pytest.fixture(scope="session")
def synthetic_cohort(tmp_path_factory):
    """Standard synthetic cohort: canonical IDs, dot decimal, comma separator."""
    root = tmp_path_factory.mktemp("cohort_std")
    return make_synthetic_cohort(root)


@pytest.fixture(scope="session")
def synthetic_cohort_locale(tmp_path_factory):
    """Synthetic cohort with European locale: semicolon separator, comma decimal."""
    root = tmp_path_factory.mktemp("cohort_locale")
    return make_synthetic_cohort(root, locale_quirks=True)


@pytest.fixture(scope="session")
def synthetic_cohort_id_quirks(tmp_path_factory):
    """Synthetic cohort with ID normalization quirks (missing zeros, trailing space)."""
    root = tmp_path_factory.mktemp("cohort_id")
    return make_synthetic_cohort(root, id_quirks=True)
