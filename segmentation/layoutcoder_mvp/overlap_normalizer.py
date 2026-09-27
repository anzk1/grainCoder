from __future__ import annotations

from .types import ElementBox, NormalizationOperation, NormalizationResult
from .validation import prepare_element_boxes


def intersection_area(first: ElementBox, second: ElementBox) -> int:
    left = max(first.bbox[0], second.bbox[0])
    top = max(first.bbox[1], second.bbox[1])
    right = min(first.bbox[2], second.bbox[2])
    bottom = min(first.bbox[3], second.bbox[3])
    return max(0, right - left) * max(0, bottom - top)


def iou(first: ElementBox, second: ElementBox) -> float:
    intersection = intersection_area(first, second)
    union = first.area + second.area - intersection
    return intersection / union if union else 0.0


def ioa(first: ElementBox, second: ElementBox) -> float:
    return intersection_area(first, second) / first.area if first.area else 0.0


def iob(first: ElementBox, second: ElementBox) -> float:
    return intersection_area(first, second) / second.area if second.area else 0.0


def _preferred_duplicate(first: ElementBox, second: ElementBox) -> ElementBox:
    first_confidence = first.confidence if first.confidence is not None else float("-inf")
    second_confidence = second.confidence if second.confidence is not None else float("-inf")
    return min(
        (first, second),
        key=lambda box: (
            -(box.confidence if box.confidence is not None else float("-inf")),
            -box.area,
            box.element_id,
        ),
    ) if first_confidence != second_confidence or first.area != second.area else min(first, second, key=lambda box: box.element_id)


def _merged_box(first: ElementBox, second: ElementBox) -> ElementBox:
    confidence_values = [value for value in (first.confidence, second.confidence) if value is not None]
    kind = first.kind if first.kind == second.kind else "non_text"
    return ElementBox(
        element_id=min(first.element_id, second.element_id),
        bbox=(
            min(first.bbox[0], second.bbox[0]),
            min(first.bbox[1], second.bbox[1]),
            max(first.bbox[2], second.bbox[2]),
            max(first.bbox[3], second.bbox[3]),
        ),
        kind=kind,
        confidence=max(confidence_values) if confidence_values else None,
        source=first.source if first.source == second.source else "normalized",
    )


def normalize_overlaps(
    boxes: list[ElementBox],
    image_width: int,
    image_height: int,
    containment_threshold: float = 0.95,
    duplicate_iou_threshold: float = 0.90,
    max_iterations: int | None = None,
) -> NormalizationResult:
    if not 0 <= containment_threshold <= 1 or not 0 <= duplicate_iou_threshold <= 1:
        raise ValueError("overlap thresholds must be between 0 and 1")
    normalized = prepare_element_boxes(boxes, image_width, image_height)
    input_count = len(boxes)
    operations: list[NormalizationOperation] = []
    iteration_limit = max_iterations or max(1, len(normalized) * len(normalized) + 1)

    for _ in range(iteration_limit):
        changed = False
        for first_index in range(len(normalized)):
            for second_index in range(first_index + 1, len(normalized)):
                first = normalized[first_index]
                second = normalized[second_index]
                intersection = intersection_area(first, second)
                if intersection == 0:
                    continue

                if iou(first, second) >= duplicate_iou_threshold:
                    selected = _preferred_duplicate(first, second)
                    operation_type = "duplicate_keep_preferred"
                elif ioa(first, second) >= containment_threshold or iob(first, second) >= containment_threshold:
                    selected = min((first, second), key=lambda box: (-box.area, box.element_id))
                    operation_type = "containment_keep_larger"
                else:
                    selected = _merged_box(first, second)
                    operation_type = "intersection_merge_enclosing"

                operations.append(
                    NormalizationOperation(operation_type, (first.element_id, second.element_id), selected.bbox)
                )
                normalized = [
                    box for index, box in enumerate(normalized) if index not in (first_index, second_index)
                ]
                normalized.append(selected)
                normalized.sort(key=lambda box: (box.bbox[1], box.bbox[0], box.bbox[3], box.bbox[2], box.element_id))
                changed = True
                break
            if changed:
                break
        if not changed:
            break
    else:
        raise RuntimeError("overlap normalization did not converge")

    stable = [
        ElementBox(index, box.bbox, box.kind, box.confidence, box.source)
        for index, box in enumerate(normalized)
    ]
    return NormalizationResult(stable, input_count, operations)
