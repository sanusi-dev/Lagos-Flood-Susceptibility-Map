"""Factor scores, runoff composite, coastal override, and class cutoffs."""
from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.config import (
    COASTAL_HIGH_MAX_M,
    COASTAL_INFLUENCE_M,
    COASTAL_MEDIUM_MAX_M,
    DIST_HIGH_MAX_M,
    DIST_MEDIUM_MAX_M,
    HAND_HIGH_MAX_M,
    HAND_MEDIUM_MAX_M,
    IMPERVIOUS_HIGH_CLASSES,
    IMPERVIOUS_MEDIUM_CLASSES,
    RISK_CLASS_HIGH,
    RISK_CLASS_LABELS,
    RISK_CLASS_LOW,
    RISK_CLASS_MEDIUM,
    RISK_CLASS_NODATA,
    RISK_SCORE_HIGH_MIN,
    RISK_SCORE_MEDIUM_MIN,
    SLOPE_FLAT_MAX_DEG,
    SLOPE_MODERATE_MAX_DEG,
    WEIGHT_DIST,
    WEIGHT_HAND,
    WEIGHT_IMPERVIOUS,
    WEIGHT_SLOPE,
)

WORLDCOVER_LABELS = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse",
    70: "Snow / ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss / lichen",
}


def ternary_low_is_high(value: float, high_max: float, medium_max: float) -> float:
    """Map a continuous factor onto {1, 2, 3} where smaller values are more susceptible."""
    if value <= high_max:
        return 3.0
    if value <= medium_max:
        return 2.0
    return 1.0


def score_hand(hand_m: float) -> float:
    return ternary_low_is_high(hand_m, HAND_HIGH_MAX_M, HAND_MEDIUM_MAX_M)


def score_distance(dist_m: float) -> float:
    return ternary_low_is_high(dist_m, DIST_HIGH_MAX_M, DIST_MEDIUM_MAX_M)


def score_slope(slope_deg: float) -> float:
    return ternary_low_is_high(slope_deg, SLOPE_FLAT_MAX_DEG, SLOPE_MODERATE_MAX_DEG)


def score_impervious(worldcover_code: int) -> float:
    code = int(worldcover_code)
    if code in IMPERVIOUS_HIGH_CLASSES:
        return 3.0
    if code in IMPERVIOUS_MEDIUM_CLASSES:
        return 2.0
    return 1.0


def score_coastal(elevation_m: float, dist_open_water_m: float) -> float | None:
    """Return 1/2/3 when the cell is within the lagoon/ocean influence radius, else None."""
    if dist_open_water_m > COASTAL_INFLUENCE_M:
        return None
    return ternary_low_is_high(elevation_m, COASTAL_HIGH_MAX_M, COASTAL_MEDIUM_MAX_M)


def runoff_composite(
    hand_score: float,
    dist_score: float,
    slope_score: float,
    impervious_score: float,
) -> float:
    return (
        WEIGHT_HAND * hand_score
        + WEIGHT_DIST * dist_score
        + WEIGHT_SLOPE * slope_score
        + WEIGHT_IMPERVIOUS * impervious_score
    )


def final_composite(runoff: float, coastal: float | None) -> float:
    if coastal is None:
        return float(runoff)
    return float(max(runoff, coastal))


def classify_composite(score: float) -> int:
    if score >= RISK_SCORE_HIGH_MIN:
        return RISK_CLASS_HIGH
    if score >= RISK_SCORE_MEDIUM_MIN:
        return RISK_CLASS_MEDIUM
    return RISK_CLASS_LOW


def normalize_score(score: float) -> float:
    """Map the 1–3 composite onto 0–1."""
    return float(min(1.0, max(0.0, (score - 1.0) / 2.0)))


def worldcover_label(code: int) -> str:
    return WORLDCOVER_LABELS.get(int(code), f"Class {int(code)}")


def _ternary_array(values: np.ndarray, high_max: float, medium_max: float) -> np.ndarray:
    scores = np.ones_like(values, dtype=np.float32)
    scores = np.where(values <= medium_max, 2.0, scores)
    scores = np.where(values <= high_max, 3.0, scores)
    return scores


def score_hand_array(hand: np.ndarray) -> np.ndarray:
    return _ternary_array(hand, HAND_HIGH_MAX_M, HAND_MEDIUM_MAX_M)


def score_distance_array(dist: np.ndarray) -> np.ndarray:
    return _ternary_array(dist, DIST_HIGH_MAX_M, DIST_MEDIUM_MAX_M)


def score_slope_array(slope: np.ndarray) -> np.ndarray:
    return _ternary_array(slope, SLOPE_FLAT_MAX_DEG, SLOPE_MODERATE_MAX_DEG)


def score_impervious_array(worldcover: np.ndarray) -> np.ndarray:
    scores = np.ones_like(worldcover, dtype=np.float32)
    high = np.isin(worldcover, IMPERVIOUS_HIGH_CLASSES)
    medium = np.isin(worldcover, IMPERVIOUS_MEDIUM_CLASSES)
    scores = np.where(medium, 2.0, scores)
    scores = np.where(high, 3.0, scores)
    return scores


def score_coastal_array(elevation: np.ndarray, dist_open_water: np.ndarray) -> np.ndarray:
    return _ternary_array(elevation, COASTAL_HIGH_MAX_M, COASTAL_MEDIUM_MAX_M)


def compute_risk_rasters(
    hand: np.ndarray,
    dist: np.ndarray,
    slope: np.ndarray,
    worldcover: np.ndarray,
    elevation: np.ndarray,
    dist_open_water: np.ndarray,
    valid_land: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return classified raster, continuous composite, and per-factor score grids."""
    hand_s = score_hand_array(hand)
    dist_s = score_distance_array(dist)
    slope_s = score_slope_array(slope)
    imp_s = score_impervious_array(worldcover)
    coastal_s = score_coastal_array(elevation, dist_open_water)

    runoff = (
        WEIGHT_HAND * hand_s
        + WEIGHT_DIST * dist_s
        + WEIGHT_SLOPE * slope_s
        + WEIGHT_IMPERVIOUS * imp_s
    ).astype(np.float32)

    near_water = dist_open_water <= COASTAL_INFLUENCE_M
    composite = np.where(near_water, np.maximum(runoff, coastal_s), runoff).astype(np.float32)

    risk = np.full(hand.shape, RISK_CLASS_NODATA, dtype=np.uint8)
    risk = np.where(valid_land, RISK_CLASS_LOW, risk)
    risk = np.where(valid_land & (composite >= RISK_SCORE_MEDIUM_MIN), RISK_CLASS_MEDIUM, risk)
    risk = np.where(valid_land & (composite >= RISK_SCORE_HIGH_MIN), RISK_CLASS_HIGH, risk)
    composite = np.where(valid_land, composite, np.nan).astype(np.float32)

    factors = {
        "hand": hand_s,
        "distance": dist_s,
        "slope": slope_s,
        "impervious": imp_s,
        "coastal": coastal_s,
        "runoff": runoff,
        "near_water": near_water,
    }
    return risk, composite, factors


def explain_scores(
    *,
    hand_m: float,
    dist_m: float,
    slope_deg: float,
    worldcover_code: int,
    elevation_m: float,
    dist_open_water_m: float,
    risk_label: str,
) -> dict[str, Any]:
    """Build the factor breakdown and a short 'why' sentence for one point."""
    hand_s = score_hand(hand_m)
    dist_s = score_distance(dist_m)
    slope_s = score_slope(slope_deg)
    imp_s = score_impervious(worldcover_code)
    coastal_s = score_coastal(elevation_m, dist_open_water_m)
    runoff = runoff_composite(hand_s, dist_s, slope_s, imp_s)
    composite = final_composite(runoff, coastal_s)
    if composite < RISK_SCORE_MEDIUM_MIN:
        why = (
            f"{risk_label} susceptibility (score {composite:.2f} / 3). "
            "Higher ground away from drainage."
        )
    else:
        drivers: list[str] = []
        if hand_s >= 3.0:
            drivers.append(f"HAND {hand_m:.2f} m")
        if dist_s >= 3.0:
            drivers.append(f"drainage {dist_m:.0f} m away")
        if slope_s >= 3.0:
            drivers.append(f"slope {slope_deg:.2f}°")
        if imp_s >= 3.0:
            drivers.append("built-up land cover")
        if coastal_s is not None and coastal_s >= 3.0 and (coastal_s >= runoff):
            drivers.append(f"low coastal elevation {elevation_m:.2f} m")
        if not drivers:
            drivers.append("combined HAND, drainage, slope, and land cover")
        why = (
            f"{risk_label} susceptibility (score {composite:.2f} / 3). "
            f"Main drivers: {', '.join(drivers)}."
        )
    return {
        "hand_score": hand_s,
        "distance_score": dist_s,
        "slope_score": slope_s,
        "impervious_score": imp_s,
        "coastal_score": coastal_s,
        "runoff_score": round(runoff, 3),
        "composite_score": round(composite, 3),
        "score_0_1": round(normalize_score(composite), 3),
        "landcover_label": worldcover_label(worldcover_code),
        "near_open_water": coastal_s is not None,
        "message": why,
    }


def suggest_cutoffs(
    flood_scores: list[float],
    dry_scores: list[float],
) -> dict[str, Any]:
    """Grid-search medium/high cutoffs on held-out flood vs dry-control scores."""
    if not flood_scores or not dry_scores:
        return {
            "high_min": RISK_SCORE_HIGH_MIN,
            "medium_min": RISK_SCORE_MEDIUM_MIN,
            "flood_hit_rate": None,
            "dry_low_rate": None,
        }

    best: dict[str, Any] | None = None
    for high_min in np.arange(1.90, 2.81, 0.05):
        for medium_min in np.arange(1.30, high_min - 0.04, 0.05):
            flood_hit = float(np.mean([s >= medium_min for s in flood_scores]))
            flood_high = float(np.mean([s >= high_min for s in flood_scores]))
            dry_low = float(np.mean([s < medium_min for s in dry_scores]))
            dry_false_high = float(np.mean([s >= high_min for s in dry_scores]))
            objective = flood_hit + dry_low + 0.35 * flood_high - 0.50 * dry_false_high
            candidate = {
                "high_min": round(float(high_min), 2),
                "medium_min": round(float(medium_min), 2),
                "flood_hit_rate": round(flood_hit, 3),
                "flood_high_rate": round(flood_high, 3),
                "dry_low_rate": round(dry_low, 3),
                "dry_false_high_rate": round(dry_false_high, 3),
                "objective": round(float(objective), 3),
            }
            if best is None or candidate["objective"] > best["objective"]:
                best = candidate
    assert best is not None
    return best


def class_label(code: int) -> str:
    return RISK_CLASS_LABELS.get(int(code), "Unknown")
