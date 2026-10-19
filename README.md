# Lagos Flood Susceptibility Map

Terrain-based flood susceptibility map for Lagos, Nigeria (`3.0°E–4.0°E`, `6.3°N–6.8°N`) at 30 m resolution.

The GIS pipeline scores land cells from elevation, drainage, slope, built-up land cover, and coastal/lagoon proximity. A Django and Leaflet app serves the map and a point-query API. This is a screening map of where runoff and low-lying coastal water are likely to collect.

![Lagos flood susceptibility map](data/outputs/lagos_risk_map.png)

## Stack

- Python 3.12
- Django 6
- Rasterio, GeoPandas, Whitebox, OSMnx, NumPy, SciPy
- Leaflet (web map)
- ESA WorldCover 2021 (built-up / land cover)

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Processed rasters live under `data/`. Re-run the pipeline only if you need to rebuild them.

## Run the web map

```bash
.venv/bin/python manage.py runserver 127.0.0.1:8000
```

Open http://127.0.0.1:8000/. Click the map, search a place, or query the API:

```bash
curl "http://127.0.0.1:8000/api/health/"
curl "http://127.0.0.1:8000/api/point/?lat=6.5880&lon=3.4020"
curl "http://127.0.0.1:8000/api/search/?q=Ikeja"
```

The point API returns elevation, HAND, distance to drainage, slope, land cover, factor scores (1–3), a composite score (1–3 and 0–1), and the susceptibility class.

For a non-debug process set `DJANGO_DEBUG=false`, `DJANGO_SECRET_KEY`, and `DJANGO_ALLOWED_HOSTS`.

## Tests

```bash
.venv/bin/pytest
```

Covers the classifier, raster sampling, `/api/point/` validation, `/api/health/` raster checks, place search, and the validation CSV.

Current validation run (40 sites): 24 of 25 flood-prone points land in High or Medium; all 15 dry-land controls land in Low. Festac Town is the flood-prone miss (HAND 6.9 m, 720 m from mapped drainage).

## Rebuild the GIS layers

Downloads cache in `data/raw/`. Completed outputs in `data/processed/` and `data/outputs/` are skipped unless you pass `--force`.

```bash
.venv/bin/python -m pipeline.run_all
```

Individual steps:

```bash
.venv/bin/python -m pipeline.step1_dem
.venv/bin/python -m pipeline.step2_hydrology
.venv/bin/python -m pipeline.step3_hand
.venv/bin/python -m pipeline.step5_context
.venv/bin/python -m pipeline.step4_classify
.venv/bin/python -m pipeline.step6_validate --csv data/validation_points.csv
```

1. Mosaic Copernicus GLO-30 DEM tiles and warp to `EPSG:32631` (UTM 31N) at 30 m.
2. Hydrologically condition the DEM and extract a D8 stream network.
3. Compute HAND, distance to drainage, and slope. OSM waterways are merged with Lagos System 1–6 canal trunks in `data/reference/lagos_system_canals.geojson`.
4. Warp ESA WorldCover, distance to open water, and LGA polygons.
5. Classify High / Medium / Low. Runoff score is a weighted mix of HAND, drainage distance, slope, and built-up land cover. Within 2 km of the lagoon or ocean, the cell also takes `max(runoff, coastal elevation score)`.
6. Validate against flood-prone sites and dry-land controls in `data/validation_points.csv`.

Thresholds and weights are in `pipeline/config.py`. Scoring lives in `pipeline/scoring.py` so the map, API, and tests use the same rules.

## Data

| Dataset | Source |
| --- | --- |
| Copernicus GLO-30 DEM and water-body mask | ESA / Sinergise AWS (`s3://copernicus-dem-30m`) |
| ESA WorldCover 2021 | ESA (`s3://esa-worldcover`) |
| Waterways and basemap | OpenStreetMap (Overpass / OSMnx) |
| System 1–6 canal trunks | `data/reference/lagos_system_canals.geojson` |
| LGA boundaries | geoBoundaries ADM2, clipped in `data/reference/lagos_lgas.geojson` |
| Validation sites | `data/validation_points.csv` (NIHSA, LASEMA, published studies, plus dry-land controls) |

## Limits

Copernicus GLO-30 is a 30 m DSM, so buildings and canopy raise elevations in built-up areas. HAND models gravity-driven runoff; the coastal factor is a simple elevation screen. OSM and the System 1–6 trunks are incomplete and have no channel capacity. Use this as a prototype screening tool only.
