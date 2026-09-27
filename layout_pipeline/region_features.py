from __future__ import annotations

from statistics import median

import numpy as np

from .schema import BBox, ElementNode, LayoutGroup, RegionFeatures, SeparatorLine, intersection_area


def extract_region_features(
    image_pixels: np.ndarray,
    region: BBox,
    page_size: tuple[int, int],
    elements: list[ElementNode],
    layouts: list[LayoutGroup],
    detected_lines: list[SeparatorLine],
) -> RegionFeatures:
    page_width, page_height = page_size
    left, top, right, bottom = region
    region_width = max(1, right - left)
    region_height = max(1, bottom - top)
    region_area = region_width * region_height
    region_elements = [element for element in elements if intersection_area(element.bbox, region) > 0]
    foreground = [element for element in region_elements if element.segmentation_role == "foreground"]
    text_elements = [element for element in foreground if element.is_text]
    non_text_elements = [element for element in foreground if not element.is_text]
    containers = [element for element in region_elements if element.segmentation_role != "foreground"]
    region_layouts = [layout for layout in layouts if intersection_area(layout.bbox, region) > 0]

    text_heights = [element.height / region_height for element in text_elements]
    text_centers_x = [element.center[0] for element in text_elements]
    text_centers_y = [element.center[1] for element in text_elements]
    component_area_ratios = [intersection_area(element.bbox, region) / region_area for element in non_text_elements + containers]
    foreground_coverage = sum(intersection_area(element.bbox, region) for element in foreground) / region_area
    container_coverage = sum(intersection_area(element.bbox, region) for element in containers) / region_area

    horizontal_gap, vertical_gap = _strongest_axis_gaps(foreground, region)
    horizontal_separator, vertical_separator = _strongest_detected_separators(detected_lines, region)
    repeated_size_score = _repeated_size_score(foreground)
    repeated_gap_score = _repeated_gap_score(foreground)
    color_variance, horizontal_transition, vertical_transition, edge_density = _pixel_features(
        image_pixels[top:bottom, left:right]
    )
    background_continuity = _clamp01(1.0 - color_variance * 0.55 - edge_density * 0.45)
    center_y = (top + bottom) / 2 / max(1, page_height)
    if center_y <= 0.2:
        page_position = "header"
    elif center_y >= 0.82:
        page_position = "footer"
    else:
        page_position = "content"

    return RegionFeatures(
        bbox=region,
        page_width_ratio=_clamp01(region_width / max(1, page_width)),
        page_height_ratio=_clamp01(region_height / max(1, page_height)),
        page_area_ratio=_clamp01(region_area / max(1, page_width * page_height)),
        page_position=page_position,
        foreground_count=len(foreground),
        text_count=len(text_elements),
        non_text_count=len(non_text_elements),
        text_ratio=_clamp01(len(text_elements) / max(1, len(foreground))),
        element_density=_clamp01(foreground_coverage),
        median_text_height=_clamp01(median(text_heights) if text_heights else 0.0),
        text_line_count=_text_line_count(text_elements),
        horizontal_text_spread=_normalized_spread(text_centers_x, region_width),
        vertical_text_spread=_normalized_spread(text_centers_y, region_height),
        largest_component_area_ratio=_clamp01(max(component_area_ratios, default=0.0)),
        container_coverage_ratio=_clamp01(container_coverage),
        row_group_count=sum(layout.layout_type == "row" for layout in region_layouts),
        column_group_count=sum(layout.layout_type == "column" for layout in region_layouts),
        grid_group_count=sum(layout.layout_type == "grid" for layout in region_layouts),
        repeated_size_score=repeated_size_score,
        repeated_gap_score=repeated_gap_score,
        strongest_horizontal_gap=horizontal_gap,
        strongest_vertical_gap=vertical_gap,
        strongest_horizontal_separator=horizontal_separator,
        strongest_vertical_separator=vertical_separator,
        color_variance=color_variance,
        horizontal_color_transition=horizontal_transition,
        vertical_color_transition=vertical_transition,
        background_continuity=background_continuity,
        edge_density=edge_density,
    )


def _clamp01(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)


def _normalized_spread(values: list[float], axis_length: int) -> float:
    if len(values) < 2:
        return 0.0
    return _clamp01((max(values) - min(values)) / max(1, axis_length))


def _text_line_count(text_elements: list[ElementNode]) -> int:
    if not text_elements:
        return 0
    tolerance = max(2.0, median(element.height for element in text_elements) * 0.6)
    centers = sorted(element.center[1] for element in text_elements)
    line_count = 1
    previous = centers[0]
    for center in centers[1:]:
        if center - previous > tolerance:
            line_count += 1
        previous = center
    return line_count


def _strongest_axis_gaps(elements: list[ElementNode], region: BBox) -> tuple[float, float]:
    if len(elements) < 2:
        return 0.0, 0.0
    left, top, right, bottom = region
    vertical_order = sorted(elements, key=lambda element: (element.bbox[1], element.bbox[0]))
    horizontal_order = sorted(elements, key=lambda element: (element.bbox[0], element.bbox[1]))
    horizontal_gaps = [
        max(0, second.bbox[1] - first.bbox[3])
        for first, second in zip(vertical_order, vertical_order[1:])
    ]
    vertical_gaps = [
        max(0, second.bbox[0] - first.bbox[2])
        for first, second in zip(horizontal_order, horizontal_order[1:])
    ]
    return (
        _clamp01(max(horizontal_gaps, default=0) / max(1, bottom - top)),
        _clamp01(max(vertical_gaps, default=0) / max(1, right - left)),
    )


def _strongest_detected_separators(
    detected_lines: list[SeparatorLine],
    region: BBox,
) -> tuple[float, float]:
    left, top, right, bottom = region
    horizontal = 0.0
    vertical = 0.0
    for line in detected_lines:
        if line.direction == "horizontal" and top < line.position < bottom:
            horizontal = max(horizontal, (min(right, line.end) - max(left, line.start)) / max(1, right - left))
        if line.direction == "vertical" and left < line.position < right:
            vertical = max(vertical, (min(bottom, line.end) - max(top, line.start)) / max(1, bottom - top))
    return _clamp01(horizontal), _clamp01(vertical)


def _repeated_size_score(elements: list[ElementNode]) -> float:
    if len(elements) < 2:
        return 0.0
    widths = np.asarray([element.width for element in elements], dtype=np.float32)
    heights = np.asarray([element.height for element in elements], dtype=np.float32)
    width_similarity = 1.0 - float(np.std(widths) / max(1.0, np.mean(widths)))
    height_similarity = 1.0 - float(np.std(heights) / max(1.0, np.mean(heights)))
    return _clamp01((width_similarity + height_similarity) / 2)


def _repeated_gap_score(elements: list[ElementNode]) -> float:
    if len(elements) < 3:
        return 0.0
    centers_x = sorted(element.center[0] for element in elements)
    centers_y = sorted(element.center[1] for element in elements)
    x_gaps = np.diff(centers_x)
    y_gaps = np.diff(centers_y)
    scores = []
    for gaps in (x_gaps, y_gaps):
        positive = gaps[gaps > 1]
        if len(positive) >= 2:
            scores.append(1.0 - float(np.std(positive) / max(1.0, np.mean(positive))))
    return _clamp01(max(scores, default=0.0))


def _pixel_features(region_pixels: np.ndarray) -> tuple[float, float, float, float]:
    if region_pixels.size == 0 or min(region_pixels.shape[:2]) < 2:
        return 0.0, 0.0, 0.0, 0.0
    pixels = region_pixels.astype(np.float32)
    color_variance = _clamp01(float(np.mean(np.std(pixels, axis=(0, 1)))) / 127.5)
    horizontal_transition = _clamp01(float(np.max(np.linalg.norm(np.diff(pixels.mean(axis=1), axis=0), axis=1))) / 441.673)
    vertical_transition = _clamp01(float(np.max(np.linalg.norm(np.diff(pixels.mean(axis=0), axis=0), axis=1))) / 441.673)
    grayscale = pixels.mean(axis=2)
    horizontal_edges = np.abs(np.diff(grayscale, axis=0))
    vertical_edges = np.abs(np.diff(grayscale, axis=1))
    edge_pixels = np.count_nonzero(horizontal_edges > 20) + np.count_nonzero(vertical_edges > 20)
    edge_total = horizontal_edges.size + vertical_edges.size
    return color_variance, horizontal_transition, vertical_transition, _clamp01(edge_pixels / max(1, edge_total))
