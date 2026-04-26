from __future__ import annotations

import math

import cv2
import numpy as np

from hvac_duct_annotator.models import DuctKind, DuctSegment


def _crop_gray(
    img_bgr: np.ndarray,
    seg: DuctSegment,
    px_per_pdf_point: float,
    pad: int = 8,
) -> np.ndarray | None:
    h, w = img_bgr.shape[:2]
    z = px_per_pdf_point
    x0 = max(0, int(seg.bbox.x0 * z) - pad)
    y0 = max(0, int(seg.bbox.y0 * z) - pad)
    x1 = min(w, int(seg.bbox.x1 * z) + pad)
    y1 = min(h, int(seg.bbox.y1 * z) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = img_bgr[y0:y1, x0:x1]
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def _fft_hatch_score(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    arr = gray.astype(np.float32)
    arr = arr - arr.mean()
    f = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    mag[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0
    yy, xx = np.indices(mag.shape)
    ang = np.rad2deg(np.arctan2(yy - cy, xx - cx))
    mask45 = (np.abs(np.abs(ang) - 45) < 12) | (np.abs(np.abs(ang) - 135) < 12)
    directional = float(mag[mask45].mean()) if np.any(mask45) else 0.0
    total = float(mag.mean() + 1e-6)
    return directional / total


def _hough_density_score(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=4)
    if lines is None:
        return 0.0
    diagonal = 0
    for ln in lines[:, 0]:
        x1, y1, x2, y2 = ln
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if 25 <= ang <= 65 or 115 <= ang <= 155:
            diagonal += 1
    return diagonal / max(1, len(lines))


def filter_duct_candidates(
    image_bgr: np.ndarray,
    segments: list[DuctSegment],
    px_per_pdf_point: float,
) -> list[DuctSegment]:
    filtered: list[DuctSegment] = []
    for seg in segments:
        if seg.kind == DuctKind.ROUND:
            seg.confidence = max(seg.confidence, 0.7)
            filtered.append(seg)
            continue
        gray = _crop_gray(image_bgr, seg, px_per_pdf_point)
        if gray is None:
            continue
        fft_score = _fft_hatch_score(gray)
        hough_score = _hough_density_score(gray)
        seg.confidence = max(seg.confidence, min(1.0, 0.55 * fft_score + 0.45 * hough_score))
        if fft_score >= 1.05 or hough_score >= 0.22 or seg.pixel_width and seg.pixel_width > 12:
            filtered.append(seg)
    return filtered
