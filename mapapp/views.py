"""Views for the Lagos Flood Susceptibility Map web application and point query API."""
from __future__ import annotations

from functools import lru_cache
import json
import math
from typing import Any

from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from pyproj import Transformer
import rasterio
from rasterio.windows import Window

from pipeline.config import (
    DEM_NODATA,
    DEM_UTM31N_PATH,
    DIST_DRAINAGE_UTM31N_PATH,
    HAND_UTM31N_PATH,
    LAGOS_BBOX_WGS84,
    RISK_CLASS_LABELS,
    RISK_METADATA_JSON_PATH,
    RISK_UTM31N_PATH,
    SLOPE_UTM31N_PATH,
    UTM_CRS,
    WBM_UTM31N_PATH,
    WGS84_CRS,
)

# Explicit coordinate transformer from EPSG:4326 (lon, lat) to EPSG:32631 (easting, northing in meters).
WGS84_TO_UTM31N = Transformer.from_crs(WGS84_CRS, UTM_CRS, always_xy=True)


@lru_cache(maxsize=1)
def get_raster_datasets() -> dict[str, rasterio.DatasetReader]:
    """Keep read-only rasterio handles open for fast point queries."""
    return {
        "dem": rasterio.open(DEM_UTM31N_PATH),
        "hand": rasterio.open(HAND_UTM31N_PATH),
        "dist": rasterio.open(DIST_DRAINAGE_UTM31N_PATH),
        "slope": rasterio.open(SLOPE_UTM31N_PATH),
        "risk": rasterio.open(RISK_UTM31N_PATH),
        "wbm": rasterio.open(WBM_UTM31N_PATH),
    }


def load_overlay_metadata() -> dict[str, Any]:
    if RISK_METADATA_JSON_PATH.exists():
        with open(RISK_METADATA_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    min_lon, min_lat, max_lon, max_lat = LAGOS_BBOX_WGS84
    return {
        "leaflet_bounds": [[min_lat, min_lon], [max_lat, max_lon]],
    }


def index(request: HttpRequest):
    """Render the single-page Leaflet flood susceptibility map."""
    meta = load_overlay_metadata()
    context = {
        "overlay_metadata_json": json.dumps(meta),
    }
    return render(request, "mapapp/index.html", context)


def query_point_rasters(lat: float, lon: float) -> dict[str, Any]:
    """Convert WGS84 lat/lon to EPSG:32631 and sample elevation, HAND, distance to drainage, slope, and risk."""
    easting_m, northing_m = WGS84_TO_UTM31N.transform(lon, lat)
    datasets = get_raster_datasets()
    dem_ds = datasets["dem"]
    bounds = dem_ds.bounds

    if not (bounds.left <= easting_m < bounds.right and bounds.bottom <= northing_m < bounds.top):
        return {
            "status": "out_of_bounds",
            "in_bounds": False,
            "is_water": False,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "utm_easting_m": round(easting_m, 1),
            "utm_northing_m": round(northing_m, 1),
            "crs": UTM_CRS,
            "elevation": None,
            "hand": None,
            "distance_to_drainage": None,
            "slope_deg": None,
            "risk_class": "Out of Bounds",
            "risk_code": None,
            "message": "Point lies outside the Lagos study area bounding box (lon 3.0–4.0°E, lat 6.3–6.8°N).",
        }

    row, col = dem_ds.index(easting_m, northing_m)
    if row < 0 or row >= dem_ds.height or col < 0 or col >= dem_ds.width:
        return {
            "status": "out_of_bounds",
            "in_bounds": False,
            "is_water": False,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "utm_easting_m": round(easting_m, 1),
            "utm_northing_m": round(northing_m, 1),
            "crs": UTM_CRS,
            "elevation": None,
            "hand": None,
            "distance_to_drainage": None,
            "slope_deg": None,
            "risk_class": "Out of Bounds",
            "risk_code": None,
            "message": "Point lies outside the raster grid extent.",
        }

    win = Window(col, row, 1, 1)
    elev_val = float(datasets["dem"].read(1, window=win)[0, 0])
    hand_val = float(datasets["hand"].read(1, window=win)[0, 0])
    dist_val = float(datasets["dist"].read(1, window=win)[0, 0])
    slope_val = float(datasets["slope"].read(1, window=win)[0, 0])
    risk_code = int(datasets["risk"].read(1, window=win)[0, 0])
    wbm_code = int(datasets["wbm"].read(1, window=win)[0, 0])

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
            "status": "water_or_nodata",
            "in_bounds": True,
            "is_water": True,
            "water_body": water_label,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "utm_easting_m": round(easting_m, 1),
            "utm_northing_m": round(northing_m, 1),
            "crs": UTM_CRS,
            "elevation": elev_clean,
            "hand": None,
            "distance_to_drainage": 0.0,
            "slope_deg": 0.0,
            "risk_class": "Water / NoData",
            "risk_code": 0,
            "message": f"Permanent water body ({water_label}). HAND runoff susceptibility applies to land cells.",
        }

    risk_label = RISK_CLASS_LABELS.get(risk_code, "Unknown")
    return {
        "status": "ok",
        "in_bounds": True,
        "is_water": False,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "utm_easting_m": round(easting_m, 1),
        "utm_northing_m": round(northing_m, 1),
        "crs": UTM_CRS,
        "elevation": round(elev_val, 2),
        "hand": round(hand_val, 2),
        "distance_to_drainage": round(dist_val, 1),
        "slope_deg": round(slope_val, 2),
        "risk_class": risk_label,
        "risk_code": risk_code,
        "message": f"{risk_label} flood susceptibility (HAND = {hand_val:.2f} m, drainage distance = {dist_val:.0f} m).",
    }


def point_query(request: HttpRequest) -> JsonResponse:
    """API endpoint: /api/point/?lat=..&lon=.."""
    lat_raw = request.GET.get("lat")
    lon_raw = request.GET.get("lon")
    if lat_raw is None or lon_raw is None:
        return JsonResponse(
            {"status": "error", "message": "Missing required query parameters 'lat' and 'lon'."},
            status=400,
        )

    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except ValueError:
        return JsonResponse(
            {"status": "error", "message": "Parameters 'lat' and 'lon' must be valid numbers."},
            status=400,
        )

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return JsonResponse(
            {"status": "error", "message": "Coordinates out of valid geographic range."},
            status=400,
        )

    result = query_point_rasters(lat=lat, lon=lon)
    return JsonResponse(result)


def health_check(request: HttpRequest) -> JsonResponse:
    return JsonResponse({"status": "ok", "project": "Lagos Flood Susceptibility Map"})
