from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DuctKind(str, Enum):
    RECT = "rect"
    ROUND = "round"


class DuctClass(str, Enum):
    SUPPLY = "supply"
    RETURN = "return"
    EXHAUST = "exhaust"
    UNCLASSIFIED = "unclassified"


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class Point(BaseModel):
    x: float
    y: float


class DimensionLabel(BaseModel):
    text: str
    normalized: str
    confidence: float = 0.0
    bbox: BBox


class DuctDimensions(BaseModel):
    shape: Literal["rect", "round"]
    width_in: float | None = None
    height_in: float | None = None
    diameter_in: float | None = None


class LengthInfo(BaseModel):
    feet: int
    inches: int
    total_inches: float


class DuctSegment(BaseModel):
    id: str
    kind: DuctKind
    bbox: BBox
    centerline: list[Point] = Field(default_factory=list)
    pixel_width: float | None = None
    pixel_length: float = 0.0
    dimensions: DuctDimensions | None = None
    label: DimensionLabel | None = None
    length: LengthInfo | None = None
    classification: DuctClass = DuctClass.UNCLASSIFIED
    confidence: float = 0.0
    geom_score: float = 0.0
    ocr_score: float = 0.0
    consistency_score: float = 0.0
    review_required: bool = False
    review_reason: str | None = None


class ScaleInfo(BaseModel):
    source: str = "default"
    expression: str = "1/4\"=1'-0\""
    inches_per_pixel: float = 0.0
    calibrated_inches_per_pixel: float | None = None


class AnnotationReport(BaseModel):
    input_path: str
    output_path: str
    scale: ScaleInfo
    segments: list[DuctSegment] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    review_queue_count: int = 0
