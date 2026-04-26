from __future__ import annotations

from dataclasses import dataclass

from hvac_duct_annotator.models import BBox


@dataclass(slots=True)
class SymbolAnchor:
    kind: str
    value: int | None
    bbox: BBox


def detect_symbol_anchors_from_text(native_text: list[tuple[str, BBox]]) -> list[SymbolAnchor]:
    """
    Lightweight anchor detector from native text spans.
    This is intentionally deterministic and robust for flattened PDFs where full
    template matching is expensive/noisy.
    """
    anchors: list[SymbolAnchor] = []
    for txt, box in native_text:
        compact = txt.replace(" ", "").upper()
        if compact.startswith(("RTU", "AHU")):
            anchors.append(SymbolAnchor(kind="equipment", value=None, bbox=box))
            continue
        if len(compact) >= 2 and compact[0] in {"A", "B", "C", "F"}:
            digits = "".join(ch for ch in compact[1:] if ch.isdigit())
            if digits:
                anchors.append(SymbolAnchor(kind=compact[0], value=int(digits), bbox=box))
    return anchors
