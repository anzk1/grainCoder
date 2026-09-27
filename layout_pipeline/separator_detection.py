from __future__ import annotations

from math import hypot

import cv2
import numpy as np

from .schema import SeparatorLine


def detect_separator_lines(image_path: str, minimum_span_ratio: float = 0.5) -> list[SeparatorLine]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"unable to read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    detector = cv2.createLineSegmentDetector(
        refine=cv2.LSD_REFINE_STD,
        scale=0.8,
        sigma_scale=0.05,
        quant=2.0,
        ang_th=22.5,
        log_eps=0,
        density_th=0.6,
        n_bins=1024,
    )
    detected = detector.detect(gray)[0]
    candidates: list[SeparatorLine] = []
    for raw_line in [] if detected is None else detected:
        x1, y1, x2, y2 = raw_line[0]
        length = hypot(x2 - x1, y2 - y1)
        if abs(y2 - y1) <= 3 and length >= width * minimum_span_ratio:
            candidates.append(SeparatorLine("horizontal", round((y1 + y2) / 2), round(min(x1, x2)), round(max(x1, x2)), "lsd"))
        elif abs(x2 - x1) <= 3 and length >= height * minimum_span_ratio:
            candidates.append(SeparatorLine("vertical", round((x1 + x2) / 2), round(min(y1, y2)), round(max(y1, y2)), "lsd"))

    deduplicated: list[SeparatorLine] = []
    for line in sorted(candidates, key=lambda item: (item.direction, item.position, item.start, item.end)):
        if any(existing.direction == line.direction and abs(existing.position - line.position) <= 5 for existing in deduplicated):
            continue
        deduplicated.append(line)
    return deduplicated
