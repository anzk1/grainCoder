from .types import BBox, ElementBox


def clip_bbox(bbox: tuple[float, float, float, float], image_width: int, image_height: int) -> BBox:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    left, top, right, bottom = (round(value) for value in bbox)
    return (
        min(max(left, 0), image_width),
        min(max(top, 0), image_height),
        min(max(right, 0), image_width),
        min(max(bottom, 0), image_height),
    )


def prepare_element_boxes(
    boxes: list[ElementBox], image_width: int, image_height: int, minimum_size: int = 2
) -> list[ElementBox]:
    prepared = []
    for box in boxes:
        clipped_bbox = clip_bbox(box.bbox, image_width, image_height)
        left, top, right, bottom = clipped_bbox
        if right - left < minimum_size or bottom - top < minimum_size:
            continue
        prepared.append(
            ElementBox(
                element_id=box.element_id,
                bbox=clipped_bbox,
                kind=box.kind,
                confidence=box.confidence,
                source=box.source,
            )
        )
    prepared.sort(key=lambda box: (box.bbox[1], box.bbox[0], box.bbox[3], box.bbox[2], box.element_id))
    return [
        ElementBox(index, box.bbox, box.kind, box.confidence, box.source)
        for index, box in enumerate(prepared)
    ]
