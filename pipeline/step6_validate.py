"""Milestone 6: Validate flood susceptibility classes against a CSV of known flood-prone locations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pyproj import Transformer
import rasterio
from rasterio.windows import Window

from pipeline.config import (
    DEM_NODATA,
    DEM_UTM31N_PATH,
    DIST_DRAINAGE_UTM31N_PATH,
    HAND_UTM31N_PATH,
    RISK_CLASS_HIGH,
    RISK_CLASS_LABELS,
    RISK_CLASS_LOW,
    RISK_CLASS_MEDIUM,
    RISK_UTM31N_PATH,
    SLOPE_UTM31N_PATH,
    UTM_CRS,
    VALIDATION_INPUT_CSV_PATH,
    VALIDATION_OUTPUT_CSV_PATH,
    VALIDATION_SUMMARY_JSON_PATH,
    WBM_UTM31N_PATH,
    WGS84_CRS,
)


def evaluate_points_csv(
    csv_path: Path = VALIDATION_INPUT_CSV_PATH,
    out_csv_path: Path = VALIDATION_OUTPUT_CSV_PATH,
    out_json_path: Path = VALIDATION_SUMMARY_JSON_PATH,
) -> dict[str, Any]:
    """Read a CSV of (lat, lon) flood-prone locations, sample EPSG:32631 rasters, and report hit rates."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}")

    # Convert WGS84 (lon, lat) to EPSG:32631 (easting, northing) in meters before raster lookup.
    transformer = Transformer.from_crs(WGS84_CRS, UTM_CRS, always_xy=True)

    with rasterio.open(DEM_UTM31N_PATH) as dem_src, \
         rasterio.open(HAND_UTM31N_PATH) as hand_src, \
         rasterio.open(DIST_DRAINAGE_UTM31N_PATH) as dist_src, \
         rasterio.open(SLOPE_UTM31N_PATH) as slp_src, \
         rasterio.open(RISK_UTM31N_PATH) as risk_src, \
         rasterio.open(WBM_UTM31N_PATH) as wbm_src:
        bounds = dem_src.bounds
        results: list[dict[str, Any]] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                lat = float(row["lat"])
                lon = float(row["lon"])
                name = row.get("name") or row.get("location") or f"Point {idx}"
                lga = row.get("lga", "")
                source = row.get("source", "")

                easting_m, northing_m = transformer.transform(lon, lat)
                in_bounds = (
                    bounds.left <= easting_m < bounds.right
                    and bounds.bottom <= northing_m < bounds.top
                )

                if not in_bounds:
                    results.append(
                        {
                            "id": row.get("id", str(idx)),
                            "name": name,
                            "lga": lga,
                            "source": source,
                            "lat": round(lat, 6),
                            "lon": round(lon, 6),
                            "utm_easting_m": round(easting_m, 1),
                            "utm_northing_m": round(northing_m, 1),
                            "elevation_m": None,
                            "hand_m": None,
                            "dist_to_drainage_m": None,
                            "slope_deg": None,
                            "risk_code": None,
                            "risk_class": "Out of Bounds",
                        }
                    )
                    continue

                r_idx, c_idx = dem_src.index(easting_m, northing_m)
                win = Window(c_idx, r_idx, 1, 1)

                elev = float(dem_src.read(1, window=win)[0, 0])
                hand = float(hand_src.read(1, window=win)[0, 0])
                dist = float(dist_src.read(1, window=win)[0, 0])
                slope = float(slp_src.read(1, window=win)[0, 0])
                risk_code = int(risk_src.read(1, window=win)[0, 0])
                wbm_code = int(wbm_src.read(1, window=win)[0, 0])

                if wbm_code in (1, 2) or risk_code == 0 or hand == DEM_NODATA:
                    risk_label = "Water / NoData"
                else:
                    risk_label = RISK_CLASS_LABELS.get(risk_code, "Unknown")

                results.append(
                    {
                        "id": row.get("id", str(idx)),
                        "name": name,
                        "lga": lga,
                        "source": source,
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "utm_easting_m": round(easting_m, 1),
                        "utm_northing_m": round(northing_m, 1),
                        "elevation_m": round(elev, 2) if elev != DEM_NODATA else None,
                        "hand_m": round(hand, 2) if hand != DEM_NODATA else None,
                        "dist_to_drainage_m": round(dist, 1) if dist != DEM_NODATA else None,
                        "slope_deg": round(slope, 2) if slope != DEM_NODATA else None,
                        "risk_code": risk_code,
                        "risk_class": risk_label,
                    }
                )

    total = len(results)
    high_count = sum(1 for r in results if r["risk_code"] == RISK_CLASS_HIGH)
    med_count = sum(1 for r in results if r["risk_code"] == RISK_CLASS_MEDIUM)
    low_count = sum(1 for r in results if r["risk_code"] == RISK_CLASS_LOW)
    other_count = total - (high_count + med_count + low_count)
    high_or_med_count = high_count + med_count

    summary = {
        "input_csv": str(csv_path),
        "total_points": total,
        "high_count": high_count,
        "medium_count": med_count,
        "low_count": low_count,
        "other_count": other_count,
        "high_share_pct": round(100.0 * high_count / total, 1) if total else 0.0,
        "medium_share_pct": round(100.0 * med_count / total, 1) if total else 0.0,
        "low_share_pct": round(100.0 * low_count / total, 1) if total else 0.0,
        "high_or_medium_count": high_or_med_count,
        "high_or_medium_share_pct": round(100.0 * high_or_med_count / total, 1) if total else 0.0,
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

    print("\n--- Milestone 6 Flood Susceptibility Validation Report ---")
    print(f"Input CSV             : {csv_path}")
    print(f"Total points evaluated: {total}")
    print("-" * 102)
    print(f"{'ID':<3} {'Location Name':<38} {'Risk':<8} {'Elev(m)':>8} {'HAND(m)':>8} {'Dist(m)':>8} {'Slope(°)':>8}")
    print("-" * 102)
    for r in results:
        elev_s = f"{r['elevation_m']:.2f}" if r["elevation_m"] is not None else "N/A"
        hand_s = f"{r['hand_m']:.2f}" if r["hand_m"] is not None else "N/A"
        dist_s = f"{r['dist_to_drainage_m']:.1f}" if r["dist_to_drainage_m"] is not None else "N/A"
        slp_s = f"{r['slope_deg']:.2f}" if r["slope_deg"] is not None else "N/A"
        print(
            f"{str(r['id']):<3} {r['name'][:37]:<38} {r['risk_class']:<8} "
            f"{elev_s:>8} {hand_s:>8} {dist_s:>8} {slp_s:>8}"
        )
    print("-" * 102)
    print(f"High Risk Class       : {high_count} / {total} ({summary['high_share_pct']:.1f}%)")
    print(f"Medium Risk Class     : {med_count} / {total} ({summary['medium_share_pct']:.1f}%)")
    print(f"Low Risk Class        : {low_count} / {total} ({summary['low_share_pct']:.1f}%)")
    print(f"High + Medium Share   : {high_or_med_count} / {total} ({summary['high_or_medium_share_pct']:.1f}%)")
    print(f"Saved detailed CSV    : {out_csv_path}")
    print(f"Saved summary JSON    : {out_json_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Lagos Flood Susceptibility classes against a CSV of known flood-prone points."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=VALIDATION_INPUT_CSV_PATH,
        help="Path to input CSV with 'lat' and 'lon' columns.",
    )
    args = parser.parse_args()
    evaluate_points_csv(csv_path=args.csv)


if __name__ == "__main__":
    main()
