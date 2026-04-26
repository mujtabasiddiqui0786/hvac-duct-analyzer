from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    default = repo_root / "testset2.pdf"
    candidate = os.environ.get("HVAC_SAMPLE_PDF", str(default))
    path = Path(candidate)
    if not path.exists():
        pytest.skip(f"Sample PDF not found: {path}")
    return path
