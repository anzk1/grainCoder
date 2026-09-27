from __future__ import annotations

from .schema import RegionFeatures, RegionProfile, RegionType


_TYPE_ORDER: tuple[RegionType, ...] = (
    "hero",
    "navigation",
    "footer",
    "card_grid",
    "two_column",
    "list",
    "text_section",
    "mixed",
)


def classify_region(features: RegionFeatures) -> RegionProfile:
    repetition = (features.repeated_size_score + features.repeated_gap_score) / 2
    vertical_partition = max(features.strongest_vertical_gap, features.strongest_vertical_separator)
    horizontal_partition = max(features.strongest_horizontal_gap, features.strongest_horizontal_separator)
    vertical_dominance = vertical_partition / max(0.01, vertical_partition + horizontal_partition)
    content_band_score = max(0.0, 1.0 - abs(features.page_height_ratio - 0.55) / 0.45)
    group_total = features.row_group_count + features.column_group_count + features.grid_group_count
    low_height = 1.0 - min(1.0, features.page_height_ratio / 0.25)
    low_text_count = 1.0 - min(1.0, features.text_count / 8)

    hero_score = _weighted(
            (features.page_area_ratio, 0.20),
            (features.largest_component_area_ratio, 0.25),
            (low_text_count, 0.20),
            (features.background_continuity, 0.20),
            (1.0 - repetition, 0.15),
        )
    if features.largest_component_area_ratio < 0.2 and features.container_coverage_ratio < 0.35:
        hero_score *= 0.35

    two_column_score = _weighted(
        (vertical_partition, 0.25),
        (vertical_dominance, 0.25),
        (min(1.0, features.foreground_count / 4), 0.15),
        (content_band_score, 0.25),
        (1.0 - horizontal_partition, 0.10),
    )
    if (
        features.foreground_count < 4
        or vertical_partition < 0.15
        or vertical_partition <= horizontal_partition
    ):
        two_column_score *= 0.4

    scores: dict[RegionType, float] = {
        "hero": hero_score,
        "navigation": _weighted(
            (1.0 if features.page_position == "header" else 0.0, 0.25),
            (low_height, 0.20),
            (features.horizontal_text_spread, 0.20),
            (features.text_ratio, 0.15),
            (features.background_continuity, 0.20),
        ),
        "text_section": _weighted(
            (features.text_ratio, 0.35),
            (1.0 - min(1.0, features.non_text_count / 3), 0.20),
            (features.vertical_text_spread, 0.20),
            (1.0 - horizontal_partition, 0.15),
            (features.background_continuity, 0.10),
        ),
        "card_grid": _weighted(
            (repetition, 0.35),
            (min(1.0, (features.row_group_count + features.grid_group_count) / 2), 0.30),
            (min(1.0, features.foreground_count / 6), 0.20),
            (max(horizontal_partition, vertical_partition), 0.15),
        ),
        "two_column": two_column_score,
        "list": _weighted(
            (features.repeated_gap_score, 0.30),
            (features.repeated_size_score, 0.20),
            (features.vertical_text_spread, 0.20),
            (min(1.0, features.row_group_count / 2), 0.15),
            (features.text_ratio, 0.15),
        ),
        "footer": _weighted(
            (1.0 if features.page_position == "footer" else 0.0, 0.30),
            (features.text_ratio, 0.20),
            (min(1.0, features.foreground_count / 6), 0.15),
            (min(1.0, (features.column_group_count + features.grid_group_count) / 2), 0.20),
            (features.background_continuity, 0.15),
        ),
        "mixed": _weighted(
            (min(1.0, features.foreground_count / 8), 0.25),
            (min(1.0, group_total / 3), 0.20),
            (1.0 - features.text_ratio, 0.15),
            (1.0 - repetition, 0.15),
            (max(horizontal_partition, vertical_partition), 0.25),
        ),
    }
    ranked = sorted(scores.items(), key=lambda item: (-item[1], _TYPE_ORDER.index(item[0])))
    region_type, winning_score = ranked[0]
    if winning_score < 0.42:
        region_type = "mixed"
        winning_score = scores["mixed"]
    runner_up = max(score for candidate_type, score in ranked if candidate_type != region_type)
    confidence = max(0.0, min(1.0, winning_score * 0.7 + max(0.0, winning_score - runner_up) * 0.3))
    return RegionProfile(
        region_type,
        round(confidence, 6),
        features,
        _classification_reasons(region_type, features, repetition, horizontal_partition, vertical_partition),
    )


def _weighted(*terms: tuple[float, float]) -> float:
    return max(0.0, min(1.0, sum(value * weight for value, weight in terms)))


def _classification_reasons(
    region_type: RegionType,
    features: RegionFeatures,
    repetition: float,
    horizontal_partition: float,
    vertical_partition: float,
) -> tuple[str, ...]:
    reasons = [f"page_position:{features.page_position}"]
    if features.text_ratio >= 0.6:
        reasons.append("text_dominant")
    if features.largest_component_area_ratio >= 0.25:
        reasons.append("large_component")
    if features.background_continuity >= 0.7:
        reasons.append("continuous_background")
    if repetition >= 0.65:
        reasons.append("repeated_layout")
    if vertical_partition >= 0.35:
        reasons.append("vertical_partition")
    if horizontal_partition >= 0.35:
        reasons.append("horizontal_partition")
    if region_type == "mixed" and len(reasons) == 1:
        reasons.append("no_specialized_profile")
    return tuple(reasons)
