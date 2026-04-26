from __future__ import annotations

import numpy as np

from hvac_duct_annotator.detect import _fft_hatch_score


def test_fft_score_nonnegative() -> None:
    img = np.zeros((64, 64), dtype=np.uint8)
    score = _fft_hatch_score(img)
    assert score >= 0.0
