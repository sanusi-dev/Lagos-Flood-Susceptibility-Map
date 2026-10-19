"""Views for the Lagos Flood Susceptibility Map web application and APIs."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
import requests

from pipeline.config import (
    LAGOS_BBOX_WGS84,
    PLACES_GAZETTEER_PATH,
    REQUIRED_WEB_RASTERS,
    RISK_METADATA_JSON_PATH,
    VALIDATION_INPUT_CSV_PATH,
)
from pipeline.sample import sample_point


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

    try:
        result = sample_point(lat=lat, lon=lon)
    except FileNotFoundError as exc:
        return JsonResponse({"status": "error", "message": str(exc)}, status=503)
    return JsonResponse(result)


def health_check(request: HttpRequest) -> JsonResponse:
    missing = [path.name for path in REQUIRED_WEB_RASTERS if not path.exists()]
    payload = {
        "status": "ok" if not missing else "degraded",
        "project": "Lagos Flood Susceptibility Map",
        "missing_rasters": missing,
    }
    return JsonResponse(payload, status=200 if not missing else 503)


@lru_cache(maxsize=1)
def load_gazetteer() -> list[dict[str, Any]]:
    if not PLACES_GAZETTEER_PATH.exists():
        return []
    with open(PLACES_GAZETTEER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _gazetteer_search(query: str) -> list[dict[str, Any]]:
    needle = query.strip().lower()
    hits: list[dict[str, Any]] = []
    for place in load_gazetteer():
        name = str(place.get("name", ""))
        lga = str(place.get("lga", ""))
        if needle in name.lower() or needle in lga.lower():
            hits.append(
                {
                    "name": name,
                    "lat": place["lat"],
                    "lon": place["lon"],
                    "kind": place.get("kind", "place"),
                    "lga": lga,
                    "source": "gazetteer",
                }
            )
    return hits


def _nominatim_search(query: str) -> list[dict[str, Any]]:
    min_lon, min_lat, max_lon, max_lat = LAGOS_BBOX_WGS84
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={
            "q": query,
            "format": "json",
            "limit": 5,
            "viewbox": f"{min_lon},{max_lat},{max_lon},{min_lat}",
            "bounded": 1,
        },
        headers={"User-Agent": "LagosFloodSusceptibilityMap/1.0"},
        timeout=8,
    )
    response.raise_for_status()
    hits = []
    for item in response.json():
        hits.append(
            {
                "name": item.get("display_name", query),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "kind": item.get("type", "nominatim"),
                "lga": "",
                "source": "nominatim",
            }
        )
    return hits


def search_places(request: HttpRequest) -> JsonResponse:
    query = (request.GET.get("q") or "").strip()
    if len(query) < 2:
        return JsonResponse({"results": []})

    results = _gazetteer_search(query)
    if len(results) < 3:
        try:
            for hit in _nominatim_search(query):
                if not any(
                    abs(hit["lat"] - existing["lat"]) < 1e-4 and abs(hit["lon"] - existing["lon"]) < 1e-4
                    for existing in results
                ):
                    results.append(hit)
        except (requests.RequestException, ValueError, KeyError):
            pass
    return JsonResponse({"results": results[:8]})


def validation_points(request: HttpRequest) -> JsonResponse:
    if not VALIDATION_INPUT_CSV_PATH.exists():
        return JsonResponse({"points": []})
    import csv

    points = []
    with open(VALIDATION_INPUT_CSV_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            points.append(
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "lga": row.get("lga"),
                    "site_type": row.get("site_type", "flood_prone"),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "source": row.get("source", ""),
                }
            )
    return JsonResponse({"points": points})
