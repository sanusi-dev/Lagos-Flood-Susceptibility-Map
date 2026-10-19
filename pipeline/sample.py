"""Shared WGS84 → UTM raster sampling used by the API and validation."""
from __future__ import annotations

import atexit
import math
from pathlib import Path
from typing import Any

from pyproj import Transformer
import rasterio
from rasterio.windows import Window

from pipeline.config import (
    DEM_NODATA,
    DEM_UTM31N_PATH,
    DIST_DRAINAGE_UTM31N_PATH,
    DIST_OPEN_WATER_UTM31N_PATH,
    HAND_UTM31N_PATH,
    RISK_UTM31N_PATH,
    SLOPE_UTM31N_PATH,
    UTM_CRS,
    WBM_UTM31N_PATH,
    WGS84_CRS,
    WORLDCOVER_UTM31N_PATH,
)
from pipeline.scoring import class_label, explain_scores

LAYER_PATHS = {
    "dem": DEM_UTM31N_PATH,
    "hand": HAND_UTM31N_PATH,
    "dist": DIST_DRAINAGE_UTM31N_PATH,
    "slope": SLOPE_UTM31N_PATH,
    "risk": RISK_UTM31N_PATH,
    "wbm": WBM_UTM31N_PATH,
    "worldcover": WORLDCOVER_UTM31N_PATH,
    "dist_open_water": DIST_OPEN_WATER_UTM31N_PATH,
}

_WGS84_TO_UTM31N = Transformer.from_crs(WGS84_CRS, UTM_CRS, always_xy=True)
_STACK: RasterStack | None = None


class RasterStack:
    """Open a set of aligned GeoTIFFs once and sample 1×1 windows."""

    def __init__(self, paths: dict[str, Path] | None = None) -> None:
        self.paths = dict(paths or LAYER_PATHS)
        self.datasets: dict[str, rasterio.DatasetReader] = {}

    def open(self) -> None:
        missing = [f"{name}: {path}" for name, path in self.paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Required rasters are missing. Run the GIS pipeline first. Missing: "
                + "; ".join(missing)
            )
        self.close()
        self.datasets = {name: rasterio.open(path) for name, path in self.paths.items()}

    def close(self) -> None:
        for ds in self.datasets.values():
            try:
                ds.close()
            except Exception:
                pass
        self.datasets = {}

    def sample(self, lat: float, lon: float) -> dict[str, Any]:
        if not self.datasets:
            self.open()
        easting_m, northing_m = _WGS84_TO_UTM31N.transform(lon, lat)
        dem_ds = self.datasets["dem"]
        bounds = dem_ds.bounds
        base = {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "utm_easting_m": round(easting_m, 1),
            "utm_northing_m": round(northing_m, 1),
            "crs": UTM_CRS,
        }

        if not (bounds.left <= easting_m < bounds.right and bounds.bottom <= northing_m < bounds.top):
            return _empty_result(
                **base,
                status="out_of_bounds",
                in_bounds=False,
                is_water=False,
                risk_class="Out of Bounds",
                message="Point lies outside the Lagos study area bounding box (lon 3.0–4.0°E, lat 6.3–6.8°N).",
            )

        row, col = dem_ds.index(easting_m, northing_m)
        if row < 0 or row >= dem_ds.height or col < 0 or col >= dem_ds.width:
            return _empty_result(
                **base,
                status="out_of_bounds",
                in_bounds=False,
                is_water=False,
                risk_class="Out of Bounds",
                message="Point lies outside the raster grid extent.",
            )

        win = Window(col, row, 1, 1)
        elev_val = float(self.datasets["dem"].read(1, window=win)[0, 0])
        hand_val = float(self.datasets["hand"].read(1, window=win)[0, 0])
        dist_val = float(self.datasets["dist"].read(1, window=win)[0, 0])
        slope_val = float(self.datasets["slope"].read(1, window=win)[0, 0])
        risk_code = int(self.datasets["risk"].read(1, window=win)[0, 0])
        wbm_code = int(self.datasets["wbm"].read(1, window=win)[0, 0])
        cover_code = int(self.datasets["worldcover"].read(1, window=win)[0, 0])
        dist_water = float(self.datasets["dist_open_water"].read(1, window=win)[0, 0])

        is_open_water = wbm_code in (1, 2)
        is_nodata = (
            risk_code == 0
            or hand_val == DEM_NODATA
            or dist_val == DEM_NODATA
            or not math.isfinite(hand_val)
        )

        if is_open_water or is_nodata:
            if wbm_code == 1:
                water_label = "Lagos Lagoon / Tidal Water" if lat > 6.43 else "Atlantic Ocean"
            elif wbm_code == 2:
                water_label = "Inland Lagoon / Lake"
            else:
                water_label = "NoData"
            elev_clean = round(elev_val, 2) if (elev_val != DEM_NODATA and math.isfinite(elev_val)) else 0.0
            return {
                **base,
                "status": "water_or_nodata",
                "in_bounds": True,
                "is_water": True,
                "water_body": water_label,
                "elevation": elev_clean,
                "hand": None,
                "distance_to_drainage": 0.0,
                "slope_deg": 0.0,
                "landcover_code": cover_code,
                "landcover_label": "Water",
                "dist_open_water_m": 0.0,
                "hand_score": None,
                "distance_score": None,
                "slope_score": None,
                "impervious_score": None,
                "coastal_score": None,
                "runoff_score": None,
                "composite_score": None,
                "score_0_1": None,
                "risk_class": "Water / NoData",
                "risk_code": 0,
                "message": f"Permanent water body ({water_label}). HAND runoff susceptibility applies to land cells.",
            }

        breakdown = explain_scores(
            hand_m=hand_val,
            dist_m=dist_val,
            slope_deg=slope_val,
            worldcover_code=cover_code,
            elevation_m=elev_val,
            dist_open_water_m=dist_water,
            risk_label=class_label(risk_code),
        )
        return {
            **base,
            "status": "ok",
            "in_bounds": True,
            "is_water": False,
            "elevation": round(elev_val, 2),
            "hand": round(hand_val, 2),
            "distance_to_drainage": round(dist_val, 1),
            "slope_deg": round(slope_val, 2),
            "landcover_code": cover_code,
            "landcover_label": breakdown["landcover_label"],
            "dist_open_water_m": round(dist_water, 1),
            "hand_score": breakdown["hand_score"],
            "distance_score": breakdown["distance_score"],
            "slope_score": breakdown["slope_score"],
            "impervious_score": breakdown["impervious_score"],
            "coastal_score": breakdown["coastal_score"],
            "runoff_score": breakdown["runoff_score"],
            "composite_score": breakdown["composite_score"],
            "score_0_1": breakdown["score_0_1"],
            "near_open_water": breakdown["near_open_water"],
            "risk_class": class_label(risk_code),
            "risk_code": risk_code,
            "message": breakdown["message"],
        }


def _empty_result(
    *,
    status: str,
    in_bounds: bool,
    is_water: bool,
    risk_class: str,
    message: str,
    **base: Any,
) -> dict[str, Any]:
    return {
        **base,
        "status": status,
        "in_bounds": in_bounds,
        "is_water": is_water,
        "elevation": None,
        "hand": None,
        "distance_to_drainage": None,
        "slope_deg": None,
        "landcover_code": None,
        "landcover_label": None,
        "dist_open_water_m": None,
        "hand_score": None,
        "distance_score": None,
        "slope_score": None,
        "impervious_score": None,
        "coastal_score": None,
        "runoff_score": None,
        "composite_score": None,
        "score_0_1": None,
        "risk_class": risk_class,
        "risk_code": None,
        "message": message,
    }


def get_stack() -> RasterStack:
    global _STACK
    if _STACK is None:
        _STACK = RasterStack()
        _STACK.open()
        atexit.register(close_stack)
    return _STACK


def close_stack() -> None:
    global _STACK
    if _STACK is not None:
        _STACK.close()
        _STACK = None


def sample_point(lat: float, lon: float, stack: RasterStack | None = None) -> dict[str, Any]:
    active = stack if stack is not None else get_stack()
    return active.sample(lat, lon)
