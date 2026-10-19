from pathlib import Path

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin

from pipeline.config import DEM_NODATA, UTM_CRS
from pipeline.sample import RasterStack
from pipeline.scoring import classify_composite, final_composite, runoff_composite


LAT = 6.55
LON = 3.40


def _write(path: Path, array: np.ndarray, transform, dtype: str, nodata) -> None:
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": UTM_CRS,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def _stack_dir(tmp_path: Path) -> tuple[RasterStack, float, float]:
    transformer = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
    easting, northing = transformer.transform(LON, LAT)
    transform = from_origin(easting - 2.5 * 30.0, northing + 2.5 * 30.0, 30.0, 30.0)

    dem = np.full((5, 5), 12.0, dtype=np.float32)
    hand = np.full((5, 5), 8.0, dtype=np.float32)
    dist = np.full((5, 5), 900.0, dtype=np.float32)
    slope = np.full((5, 5), 4.0, dtype=np.float32)
    cover = np.full((5, 5), 10, dtype=np.uint8)
    dow = np.full((5, 5), 5000.0, dtype=np.float32)
    wbm = np.zeros((5, 5), dtype=np.uint8)
    risk = np.ones((5, 5), dtype=np.uint8)

    # Center pixel: flood-like. Right-of-center: open water.
    dem[2, 2] = 3.0
    hand[2, 2] = 1.2
    dist[2, 2] = 80.0
    slope[2, 2] = 0.8
    cover[2, 2] = 50
    dow[2, 2] = 400.0
    risk[2, 2] = 3

    wbm[2, 3] = 1
    risk[2, 3] = 0
    hand[2, 3] = DEM_NODATA

    paths = {
        "dem": tmp_path / "dem.tif",
        "hand": tmp_path / "hand.tif",
        "dist": tmp_path / "dist.tif",
        "slope": tmp_path / "slope.tif",
        "risk": tmp_path / "risk.tif",
        "wbm": tmp_path / "wbm.tif",
        "worldcover": tmp_path / "cover.tif",
        "dist_open_water": tmp_path / "dow.tif",
    }
    _write(paths["dem"], dem, transform, "float32", DEM_NODATA)
    _write(paths["hand"], hand, transform, "float32", DEM_NODATA)
    _write(paths["dist"], dist, transform, "float32", DEM_NODATA)
    _write(paths["slope"], slope, transform, "float32", DEM_NODATA)
    _write(paths["worldcover"], cover, transform, "uint8", 0)
    _write(paths["dist_open_water"], dow, transform, "float32", DEM_NODATA)
    _write(paths["wbm"], wbm, transform, "uint8", 255)
    _write(paths["risk"], risk, transform, "uint8", 0)

    stack = RasterStack(paths)
    stack.open()
    return stack, easting, northing


def test_sample_land_cell_returns_factor_scores(tmp_path: Path):
    stack, _, _ = _stack_dir(tmp_path)
    try:
        result = stack.sample(LAT, LON)
    finally:
        stack.close()
    assert result["status"] == "ok"
    assert result["in_bounds"] is True
    assert result["hand_score"] == 3.0
    assert result["impervious_score"] == 3.0
    assert result["composite_score"] is not None
    expected = classify_composite(
        final_composite(
            runoff_composite(
                result["hand_score"],
                result["distance_score"],
                result["slope_score"],
                result["impervious_score"],
            ),
            result["coastal_score"],
        )
    )
    assert expected == 3
    assert result["risk_class"] == "High"


def test_sample_out_of_bounds(tmp_path: Path):
    stack, _, _ = _stack_dir(tmp_path)
    try:
        result = stack.sample(6.9, 3.5)
    finally:
        stack.close()
    assert result["status"] == "out_of_bounds"
    assert result["risk_class"] == "Out of Bounds"


def test_sample_water_cell(tmp_path: Path):
    stack, easting, northing = _stack_dir(tmp_path)
    transformer = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting + 30.0, northing)
    try:
        result = stack.sample(lat, lon)
    finally:
        stack.close()
    assert result["status"] == "water_or_nodata"
    assert result["is_water"] is True
