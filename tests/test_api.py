from pathlib import Path

from django.test import Client

from mapapp import views


def test_point_query_requires_params():
    client = Client()
    response = client.get("/api/point/")
    assert response.status_code == 400
    assert response.json()["status"] == "error"


def test_point_query_rejects_non_numeric():
    client = Client()
    response = client.get("/api/point/", {"lat": "x", "lon": "3.4"})
    assert response.status_code == 400


def test_point_query_rejects_geographic_range():
    client = Client()
    response = client.get("/api/point/", {"lat": "95", "lon": "3.4"})
    assert response.status_code == 400


def test_point_query_ok_uses_sampler(monkeypatch):
    client = Client()

    def fake_sample(lat, lon):
        return {
            "status": "ok",
            "lat": lat,
            "lon": lon,
            "risk_class": "High",
            "composite_score": 2.7,
            "hand_score": 3.0,
            "distance_score": 3.0,
            "slope_score": 2.0,
            "impervious_score": 3.0,
            "coastal_score": None,
        }

    monkeypatch.setattr(views, "sample_point", fake_sample)
    response = client.get("/api/point/", {"lat": "6.55", "lon": "3.40"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_class"] == "High"
    assert payload["hand_score"] == 3.0


def test_health_ok_when_rasters_exist(monkeypatch, tmp_path: Path):
    raster = tmp_path / "dem.tif"
    raster.write_bytes(b"ok")
    monkeypatch.setattr(views, "REQUIRED_WEB_RASTERS", (raster,))
    client = Client()
    response = client.get("/api/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_degraded_when_rasters_missing(monkeypatch, tmp_path: Path):
    missing = tmp_path / "missing.tif"
    monkeypatch.setattr(views, "REQUIRED_WEB_RASTERS", (missing,))
    client = Client()
    response = client.get("/api/health/")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert "missing.tif" in response.json()["missing_rasters"]


def test_search_gazetteer_ikeja():
    client = Client()
    response = client.get("/api/search/", {"q": "Ikeja"})
    assert response.status_code == 200
    names = [hit["name"] for hit in response.json()["results"]]
    assert any("Ikeja" in name for name in names)


def test_validation_points_include_site_types():
    client = Client()
    response = client.get("/api/validation/")
    assert response.status_code == 200
    points = response.json()["points"]
    types = {pt["site_type"] for pt in points}
    assert "flood_prone" in types
    assert "dry_control" in types
    assert len(points) >= 30
