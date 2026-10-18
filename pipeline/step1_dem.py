"""Milestone 1: Download Copernicus GLO-30 DEM tiles, mosaic/clip, reproject to EPSG:32631, and render hillshade."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
import requests

from pipeline.config import (
    COPERNICUS_S3_BASE_URL,
    COPERNICUS_TILES,
    DEM_CLIPPED_WGS84_PATH,
    DEM_NODATA,
    DEM_UTM31N_PATH,
    HILLSHADE_PNG_PATH,
    HILLSHADE_UTM31N_PATH,
    LAGOS_BBOX_WGS84,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_RESOLUTION_M,
    UTM_CRS,
    WGS84_CRS,
)


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def download_copernicus_tiles(force: bool = False) -> list[Path]:
    """Download Copernicus GLO-30 COG tiles from the public AWS S3 bucket if not cached."""
    downloaded_paths: list[Path] = []
    for tile_name in COPERNICUS_TILES:
        filename = f"{tile_name}.tif"
        local_path = RAW_DIR / filename
        url = f"{COPERNICUS_S3_BASE_URL}/{tile_name}/{filename}"

        if local_path.exists() and local_path.stat().st_size > 0 and not force:
            print(f"[cache] Using existing DEM tile: {local_path} ({local_path.stat().st_size / 1e6:.2f} MB)")
            downloaded_paths.append(local_path)
            continue

        print(f"[download] Fetching {url} ...")
        response = requests.get(url, stream=True, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download {url} (HTTP {response.status_code}). "
                "Check network connectivity or S3 bucket availability."
            )

        temp_path = local_path.with_suffix(".tif.part")
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        temp_path.replace(local_path)
        print(f"[download] Saved {local_path} ({local_path.stat().st_size / 1e6:.2f} MB)")
        downloaded_paths.append(local_path)

    return downloaded_paths


def mosaic_and_clip_wgs84(tile_paths: list[Path], force: bool = False) -> Path:
    """Mosaic the raw Copernicus tiles and clip to the Lagos bounding box in EPSG:4326."""
    if DEM_CLIPPED_WGS84_PATH.exists() and not force:
        print(f"[cache] Using existing clipped WGS84 DEM: {DEM_CLIPPED_WGS84_PATH}")
        return DEM_CLIPPED_WGS84_PATH

    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic_arr, mosaic_transform = merge(
            datasets,
            bounds=LAGOS_BBOX_WGS84,
            nodata=DEM_NODATA,
        )
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic_arr.shape[1],
            width=mosaic_arr.shape[2],
            count=1,
            dtype="float32",
            crs=CRS.from_string(WGS84_CRS),
            transform=mosaic_transform,
            nodata=DEM_NODATA,
            compress="deflate",
        )
        with rasterio.open(DEM_CLIPPED_WGS84_PATH, "w", **profile) as dst:
            dst.write(mosaic_arr.astype(np.float32))
    finally:
        for ds in datasets:
            ds.close()

    print(f"[process] Saved mosaicked and clipped WGS84 DEM: {DEM_CLIPPED_WGS84_PATH}")
    return DEM_CLIPPED_WGS84_PATH


def reproject_dem_to_utm(tile_paths: list[Path], force: bool = False) -> Path:
    """Reproject the DEM to EPSG:32631 (UTM Zone 31N) on an exact 30m metric grid."""
    if DEM_UTM31N_PATH.exists() and not force:
        print(f"[cache] Using existing UTM 31N DEM: {DEM_UTM31N_PATH}")
        return DEM_UTM31N_PATH

    src_crs = CRS.from_string(WGS84_CRS)
    dst_crs = CRS.from_string(UTM_CRS)

    # Pad the geographic merge window by 0.02 degrees (~2.2 km) before warping to UTM.
    # Reason: UTM Zone 31N meridian convergence rotates the geographic bounding box slightly;
    # warping from a padded source prevents triangular NoData slivers along the UTM raster edges.
    min_lon, min_lat, max_lon, max_lat = LAGOS_BBOX_WGS84
    padded_bbox = (
        max(3.0, min_lon - 0.02),
        min_lat - 0.02,
        max_lon + 0.02,
        max_lat + 0.02,
    )

    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        padded_mosaic, padded_transform = merge(
            datasets,
            bounds=padded_bbox,
            nodata=DEM_NODATA,
        )
    finally:
        for ds in datasets:
            ds.close()

    # Compute target bounds in EPSG:32631 and snap dimensions to clean 30m pixels.
    utm_left, utm_bottom, utm_right, utm_top = transform_bounds(
        src_crs, dst_crs, min_lon, min_lat, max_lon, max_lat, densify_pts=21
    )
    utm_left = math.ceil((utm_left - 1e-6) / TARGET_RESOLUTION_M) * TARGET_RESOLUTION_M
    utm_bottom = math.floor((utm_bottom + 1e-6) / TARGET_RESOLUTION_M) * TARGET_RESOLUTION_M
    utm_right = math.ceil((utm_right - 1e-6) / TARGET_RESOLUTION_M) * TARGET_RESOLUTION_M
    utm_top = math.ceil((utm_top - 1e-6) / TARGET_RESOLUTION_M) * TARGET_RESOLUTION_M

    dst_width = int(round((utm_right - utm_left) / TARGET_RESOLUTION_M))
    dst_height = int(round((utm_top - utm_bottom) / TARGET_RESOLUTION_M))
    dst_transform = from_bounds(utm_left, utm_bottom, utm_right, utm_top, dst_width, dst_height)

    dst_arr = np.full((dst_height, dst_width), DEM_NODATA, dtype=np.float32)

    # Reason for bilinear resampling: nearest-neighbor on continuous elevation creates stepped
    # contour artifacts that distort slope and D8 flow directions in flat coastal terrain.
    reproject(
        source=padded_mosaic[0].astype(np.float32),
        destination=dst_arr,
        src_transform=padded_transform,
        src_crs=src_crs,
        src_nodata=DEM_NODATA,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=DEM_NODATA,
        resampling=Resampling.bilinear,
    )

    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": DEM_NODATA,
        "compress": "deflate",
    }
    with rasterio.open(DEM_UTM31N_PATH, "w", **profile) as dst:
        dst.write(dst_arr, 1)

    print(f"[process] Saved reprojected EPSG:32631 DEM: {DEM_UTM31N_PATH}")
    return DEM_UTM31N_PATH


def compute_hillshade(
    dem_utm_path: Path,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    z_factor: float = 5.0,
    force: bool = False,
) -> tuple[Path, Path]:
    """Compute analytical hillshade in EPSG:32631 (metric X/Y/Z) and save GeoTIFF + PNG preview."""
    if HILLSHADE_UTM31N_PATH.exists() and HILLSHADE_PNG_PATH.exists() and not force:
        print(f"[cache] Using existing hillshade image: {HILLSHADE_PNG_PATH}")
        return HILLSHADE_UTM31N_PATH, HILLSHADE_PNG_PATH

    with rasterio.open(dem_utm_path) as src:
        dem = src.read(1)
        nodata = src.nodata
        bounds = src.bounds
        res_x, res_y = src.res
        profile = src.profile.copy()

    valid_mask = (dem != nodata) & np.isfinite(dem)
    dem_filled = np.where(valid_mask, dem, 0.0)

    # Reason for z_factor=5.0: Lagos coastal plain has low relief (mostly 0-90 m over ~110 km).
    # Vertical exaggeration reveals subtle barrier beach ridges, river floodplains, and creeks.
    dy, dx = np.gradient(dem_filled * z_factor, res_y, res_x)
    slope_rad = np.arctan(np.hypot(dx, dy))
    aspect_rad = np.arctan2(dy, -dx)

    azimuth_rad = np.deg2rad(azimuth_deg)
    zenith_rad = np.deg2rad(90.0 - altitude_deg)

    shaded = (
        np.cos(zenith_rad) * np.cos(slope_rad)
        + np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )
    hillshade = np.clip(255.0 * shaded, 0.0, 255.0).astype(np.uint8)
    hillshade[~valid_mask] = 0

    hs_profile = profile.copy()
    hs_profile.update(dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(HILLSHADE_UTM31N_PATH, "w", **hs_profile) as dst:
        dst.write(hillshade, 1)

    # Render a diagnostic figure combining hillshade and hypsometric tinting.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), dpi=150, constrained_layout=True)
    extent_km = [
        bounds.left / 1000.0,
        bounds.right / 1000.0,
        bounds.bottom / 1000.0,
        bounds.top / 1000.0,
    ]

    ax0 = axes[0]
    ax0.imshow(hillshade, cmap="gray", extent=extent_km, origin="upper", vmin=40, vmax=245)
    ax0.set_title("Copernicus GLO-30 Hillshade (EPSG:32631, z=5x)", fontsize=12, fontweight="bold")
    ax0.set_xlabel("UTM Zone 31N Easting (km)")
    ax0.set_ylabel("UTM Zone 31N Northing (km)")

    ax1 = axes[1]
    dem_plot = np.ma.masked_where(~valid_mask, dem)
    ax1.imshow(hillshade, cmap="gray", extent=extent_km, origin="upper", vmin=40, vmax=245)
    im = ax1.imshow(
        dem_plot,
        cmap="terrain",
        extent=extent_km,
        origin="upper",
        vmin=-2.0,
        vmax=90.0,
        alpha=0.58,
    )
    water_mask = np.ma.masked_where(~(valid_mask & (dem <= 0.0)), np.ones_like(dem))
    ax1.imshow(water_mask, cmap="Blues", extent=extent_km, origin="upper", vmin=0, vmax=1.5, alpha=0.65)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label("Elevation (m, EGM2008)")
    ax1.set_title("Lagos Elevation + Hillshade (EPSG:32631, 30m Grid)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("UTM Zone 31N Easting (km)")
    ax1.set_ylabel("UTM Zone 31N Northing (km)")

    fig.savefig(HILLSHADE_PNG_PATH)
    plt.close(fig)

    print(f"[process] Saved hillshade GeoTIFF: {HILLSHADE_UTM31N_PATH}")
    print(f"[process] Saved hillshade PNG image: {HILLSHADE_PNG_PATH}")
    return HILLSHADE_UTM31N_PATH, HILLSHADE_PNG_PATH


def print_summary(dem_utm_path: Path) -> None:
    """Print key raster metadata and elevation summary statistics."""
    with rasterio.open(dem_utm_path) as src:
        dem = src.read(1)
        valid = dem[(dem != src.nodata) & np.isfinite(dem)]
        print("\n--- Milestone 1 DEM Summary ---")
        print(f"File path      : {dem_utm_path}")
        print(f"CRS            : {src.crs}")
        print(f"Dimensions     : {src.width} cols x {src.height} rows")
        print(f"Pixel size (m) : {src.res[0]:.1f} x {src.res[1]:.1f}")
        print(
            f"UTM Bounds (m) : left={src.bounds.left:.1f}, bottom={src.bounds.bottom:.1f}, "
            f"right={src.bounds.right:.1f}, top={src.bounds.top:.1f}"
        )
        print(f"Valid cells    : {valid.size:,} / {dem.size:,} ({100.0 * valid.size / dem.size:.2f}%)")
        print(
            f"Elevation (m)  : min={valid.min():.2f}, median={np.median(valid):.2f}, "
            f"mean={valid.mean():.2f}, p95={np.percentile(valid, 95):.2f}, max={valid.max():.2f}"
        )
        water_share = 100.0 * np.sum(valid <= 0.0) / valid.size
        print(f"Cells <= 0m    : {water_share:.2f}% (Atlantic Ocean, Lagos Lagoon, creeks)")


def run_step1(force: bool = False) -> None:
    ensure_directories()
    tile_paths = download_copernicus_tiles(force=force)
    mosaic_and_clip_wgs84(tile_paths, force=force)
    dem_utm_path = reproject_dem_to_utm(tile_paths, force=force)
    compute_hillshade(dem_utm_path, force=force)
    print_summary(dem_utm_path)


if __name__ == "__main__":
    run_step1()
