"""Validate flood susceptibility against flood-prone sites and dry-land controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pipeline.config import (
    RISK_CLASS_HIGH,
    RISK_CLASS_LOW,
    RISK_CLASS_MEDIUM,
    VALIDATION_INPUT_CSV_PATH,
    VALIDATION_OUTPUT_CSV_PATH,
    VALIDATION_SUMMARY_JSON_PATH,
)
from pipeline.sample import RasterStack, sample_point
from pipeline.scoring import suggest_cutoffs


def evaluate_points_csv(
    csv_path: Path = VALIDATION_INPUT_CSV_PATH,
    out_csv_path: Path = VALIDATION_OUTPUT_CSV_PATH,
    out_json_path: Path = VALIDATION_SUMMARY_JSON_PATH,
) -> dict[str, Any]:
    """Read a CSV of (lat, lon) sites, sample rasters, and report flood vs dry-control rates."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}")

    stack = RasterStack()
    stack.open()
    results: list[dict[str, Any]] = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                lat = float(row["lat"])
                lon = float(row["lon"])
                name = row.get("name") or row.get("location") or f"Point {idx}"
                site_type = (row.get("site_type") or "flood_prone").strip()
                sampled = sample_point(lat, lon, stack=stack)
                results.append(
                    {
                        "id": row.get("id", str(idx)),
                        "name": name,
                        "lga": row.get("lga", ""),
                        "source": row.get("source", ""),
                        "site_type": site_type,
                        "lat": sampled["lat"],
                        "lon": sampled["lon"],
                        "utm_easting_m": sampled["utm_easting_m"],
                        "utm_northing_m": sampled["utm_northing_m"],
                        "elevation_m": sampled.get("elevation"),
                        "hand_m": sampled.get("hand"),
                        "dist_to_drainage_m": sampled.get("distance_to_drainage"),
                        "slope_deg": sampled.get("slope_deg"),
                        "landcover_label": sampled.get("landcover_label"),
                        "hand_score": sampled.get("hand_score"),
                        "distance_score": sampled.get("distance_score"),
                        "slope_score": sampled.get("slope_score"),
                        "impervious_score": sampled.get("impervious_score"),
                        "coastal_score": sampled.get("coastal_score"),
                        "runoff_score": sampled.get("runoff_score"),
                        "composite_score": sampled.get("composite_score"),
                        "score_0_1": sampled.get("score_0_1"),
                        "risk_code": sampled.get("risk_code"),
                        "risk_class": sampled.get("risk_class"),
                    }
                )
    finally:
        stack.close()

    def _subset(kind: str) -> list[dict[str, Any]]:
        return [r for r in results if r["site_type"] == kind]

    def _rate(rows: list[dict[str, Any]], predicate) -> float:
        if not rows:
            return 0.0
        return round(100.0 * sum(1 for r in rows if predicate(r)) / len(rows), 1)

    flood_rows = _subset("flood_prone")
    dry_rows = _subset("dry_control")
    other_rows = [r for r in results if r["site_type"] not in {"flood_prone", "dry_control"}]

    def is_high(r: dict[str, Any]) -> bool:
        return r.get("risk_code") == RISK_CLASS_HIGH

    def is_medium(r: dict[str, Any]) -> bool:
        return r.get("risk_code") == RISK_CLASS_MEDIUM

    def is_low(r: dict[str, Any]) -> bool:
        return r.get("risk_code") == RISK_CLASS_LOW

    def is_high_or_med(r: dict[str, Any]) -> bool:
        return r.get("risk_code") in {RISK_CLASS_HIGH, RISK_CLASS_MEDIUM}

    flood_scores = [float(r["composite_score"]) for r in flood_rows if r.get("composite_score") is not None]
    dry_scores = [float(r["composite_score"]) for r in dry_rows if r.get("composite_score") is not None]
    suggested = suggest_cutoffs(flood_scores, dry_scores)

    summary = {
        "input_csv": str(csv_path),
        "total_points": len(results),
        "flood_prone_count": len(flood_rows),
        "dry_control_count": len(dry_rows),
        "other_count": len(other_rows),
        "flood_prone": {
            "count": len(flood_rows),
            "high_count": sum(1 for r in flood_rows if is_high(r)),
            "medium_count": sum(1 for r in flood_rows if is_medium(r)),
            "low_count": sum(1 for r in flood_rows if is_low(r)),
            "high_or_medium_count": sum(1 for r in flood_rows if is_high_or_med(r)),
            "high_share_pct": _rate(flood_rows, is_high),
            "high_or_medium_share_pct": _rate(flood_rows, is_high_or_med),
            "low_share_pct": _rate(flood_rows, is_low),
        },
        "dry_control": {
            "count": len(dry_rows),
            "high_count": sum(1 for r in dry_rows if is_high(r)),
            "medium_count": sum(1 for r in dry_rows if is_medium(r)),
            "low_count": sum(1 for r in dry_rows if is_low(r)),
            "high_or_medium_count": sum(1 for r in dry_rows if is_high_or_med(r)),
            "low_share_pct": _rate(dry_rows, is_low),
            "high_or_medium_share_pct": _rate(dry_rows, is_high_or_med),
        },
        "suggested_cutoffs": suggested,
        "results": results,
    }

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Flood Susceptibility Validation Report ---")
    print(f"Input CSV             : {csv_path}")
    print(f"Total points          : {len(results)}  (flood-prone {len(flood_rows)}, dry-control {len(dry_rows)})")
    print("-" * 118)
    print(
        f"{'ID':<3} {'Type':<12} {'Location':<38} {'Class':<8} "
        f"{'Score':>6} {'HAND':>7} {'Dist':>7} {'Imp':>4} {'Cst':>4}"
    )
    print("-" * 118)
    for r in results:
        score_s = f"{r['composite_score']:.2f}" if r["composite_score"] is not None else "N/A"
        hand_s = f"{r['hand_m']:.2f}" if r["hand_m"] is not None else "N/A"
        dist_s = f"{r['dist_to_drainage_m']:.0f}" if r["dist_to_drainage_m"] is not None else "N/A"
        imp_s = f"{r['impervious_score']:.0f}" if r["impervious_score"] is not None else "N/A"
        cst_s = f"{r['coastal_score']:.0f}" if r["coastal_score"] is not None else "—"
        print(
            f"{str(r['id']):<3} {r['site_type']:<12} {r['name'][:37]:<38} {str(r['risk_class']):<8} "
            f"{score_s:>6} {hand_s:>7} {dist_s:>7} {imp_s:>4} {cst_s:>4}"
        )
    print("-" * 118)
    fp = summary["flood_prone"]
    dc = summary["dry_control"]
    print(
        f"Flood-prone High+Medium : {fp['high_or_medium_count']} / {fp['count']} "
        f"({fp['high_or_medium_share_pct']:.1f}%)"
    )
    print(
        f"Dry-control Low         : {dc['low_count']} / {dc['count']} "
        f"({dc['low_share_pct']:.1f}%)"
    )
    print(
        f"Suggested cutoffs       : medium>={suggested.get('medium_min')}, "
        f"high>={suggested.get('high_min')} "
        f"(flood hit {suggested.get('flood_hit_rate')}, dry low {suggested.get('dry_low_rate')})"
    )
    print(f"Saved detailed CSV      : {out_csv_path}")
    print(f"Saved summary JSON      : {out_json_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Lagos flood susceptibility against flood-prone and dry-control points."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=VALIDATION_INPUT_CSV_PATH,
        help="Path to input CSV with lat, lon, and site_type columns.",
    )
    args = parser.parse_args()
    evaluate_points_csv(csv_path=args.csv)


if __name__ == "__main__":
    main()
