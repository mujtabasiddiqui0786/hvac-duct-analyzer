from __future__ import annotations

import re

import fitz

from hvac_duct_annotator.models import BBox

DIFFUSER_RE = re.compile(r"^([ABCF])\s*[/\-]?\s*(\d{2,4})$", re.IGNORECASE)


def extract_native_text(page: fitz.Page) -> list[tuple[str, BBox]]:
    out: list[tuple[str, BBox]] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = str(span.get("text", "")).strip()
                if not txt:
                    continue
                x0, y0, x1, y1 = span["bbox"]
                out.append((txt, BBox(x0=x0, y0=y0, x1=x1, y1=y1)))
    return out


def extract_diffuser_tags(native_text: list[tuple[str, BBox]]) -> list[tuple[str, int, BBox]]:
    tags: list[tuple[str, int, BBox]] = []
    for txt, box in native_text:
        m = DIFFUSER_RE.match(txt.replace(" ", ""))
        if not m:
            continue
        tags.append((m.group(1).upper(), int(m.group(2)), box))
    return tags
