import numpy as np

from pipeline.config import (
    RISK_CLASS_HIGH,
    RISK_CLASS_LOW,
    RISK_CLASS_MEDIUM,
    RISK_SCORE_HIGH_MIN,
    RISK_SCORE_MEDIUM_MIN,
)
from pipeline.scoring import (
    classify_composite,
    compute_risk_rasters,
    final_composite,
    runoff_composite,
    score_coastal,
    score_hand,
    score_impervious,
    suggest_cutoffs,
)


def test_hand_thresholds():
    assert score_hand(1.0) == 3.0
    assert score_hand(3.0) == 2.0
    assert score_hand(8.0) == 1.0


def test_built_up_scores_high():
    assert score_impervious(50) == 3.0
    assert score_impervious(40) == 2.0
    assert score_impervious(10) == 1.0


def test_coastal_only_near_water():
    assert score_coastal(1.0, 500.0) == 3.0
    assert score_coastal(1.0, 8000.0) is None


def test_coastal_override_raises_inland_low_runoff():
    runoff = runoff_composite(1.0, 1.0, 1.0, 1.0)
    assert runoff < RISK_SCORE_MEDIUM_MIN
    assert final_composite(runoff, 3.0) == 3.0
    assert classify_composite(final_composite(runoff, 3.0)) == RISK_CLASS_HIGH


def test_inland_built_up_upland_stays_low():
    runoff = runoff_composite(1.0, 1.0, 1.0, 3.0)
    assert classify_composite(final_composite(runoff, None)) == RISK_CLASS_LOW


def test_ijegun_like_urban_point_is_medium():
    runoff = runoff_composite(1.0, 2.0, 2.0, 3.0)
    assert classify_composite(final_composite(runoff, None)) == RISK_CLASS_MEDIUM


def test_compute_risk_rasters_coastal_and_valid_mask():
    hand = np.array([[10.0, 1.0], [10.0, 10.0]], dtype=np.float32)
    dist = np.array([[900.0, 40.0], [900.0, 900.0]], dtype=np.float32)
    slope = np.array([[5.0, 0.5], [5.0, 5.0]], dtype=np.float32)
    cover = np.array([[10, 50], [10, 10]], dtype=np.uint8)
    elev = np.array([[20.0, 4.0], [1.2, 20.0]], dtype=np.float32)
    dow = np.array([[5000.0, 5000.0], [200.0, 5000.0]], dtype=np.float32)
    valid = np.array([[True, True], [True, False]])

    risk, score, factors = compute_risk_rasters(hand, dist, slope, cover, elev, dow, valid)
    assert risk[0, 1] == RISK_CLASS_HIGH
    assert risk[1, 0] == RISK_CLASS_HIGH
    assert risk[1, 1] == 0
    assert np.isnan(score[1, 1])
    assert factors["near_water"][1, 0]


def test_suggest_cutoffs_separates_flood_and_dry():
    suggested = suggest_cutoffs([2.7, 2.6, 2.2, 1.9], [1.2, 1.3, 1.4, 1.1])
    assert suggested["medium_min"] <= suggested["high_min"]
    assert suggested["flood_hit_rate"] >= 0.75
    assert suggested["dry_low_rate"] >= 0.75
    assert RISK_SCORE_HIGH_MIN > 1.0
    assert RISK_SCORE_MEDIUM_MIN > 1.0
