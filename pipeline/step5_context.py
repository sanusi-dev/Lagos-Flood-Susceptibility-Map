"""Land cover, distance to open water, and LGA overlay for the study grid."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

from pipeline.config import (
    CONTEXT_PNG_PATH,
    DEM_UTM31N_PATH,
    DIST_OPEN_WATER_UTM31N_PATH,
    HILLSHADE_UTM31N_PATH,
    LAGOS_BBOX_WGS84,
    LAGOS_LGAS_PATH,
    LAGOS_LGAS_WGS84_PATH,
    OUTPUTS_DIR,
    RAW_DIR,
    TARGET_RESOLUTION_M,
    UTM_CRS,
    WBM_UTM31N_PATH,
    WGS84_CRS,
    WORLDCOVER_S3_BASE_URL,
    WORLDCOVER_TILES,
    WORLDCOVER_UTM31N_PATH,
)


def _enable_vsicurl() -> None:
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")


def warp_worldcover(force: bool = False) -> Path:
    """Warp ESA WorldCover 2021 onto the DEM grid with nearest-neighbor resampling."""
    if WORLDCOVER_UTM31N_PATH.exists() and not force:
        print(f"[cache] Using existing WorldCover raster: {WORLDCOVER_UTM31N_PATH}")
        return WORLDCOVER_UTM31N_PATH

    _enable_vsicurl()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tile_name = WORLDCOVER_TILES[0]
    local_path = RAW_DIR / f"{tile_name}.tif"
    url = f"{WORLDCOVER_S3_BASE_URL}/{tile_name}.tif"
    source_path = str(local_path) if local_path.exists() and local_path.stat().st_size > 0 else f"/vsicurl/{url}"

    with rasterio.open(DEM_UTM31N_PATH) as dem_src:
        dst = np.zeros((dem_src.height, dem_src.width), dtype=np.uint8)
        profile = dem_src.profile.copy()
        dst_transform = dem_src.transform
        dst_crs = dem_src.crs
        dst_height, dst_width = dem_src.height, dem_src.width

    print(f"[process] Warping WorldCover from {source_path} ...")
    try:
        with rasterio.open(source_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
    except Exception as exc:
        if source_path.startswith("/vsicurl/"):
            print(f"[download] vsicurl failed ({exc}); fetching {url} to cache ...")
            import requests

            response = requests.get(url, stream=True, timeout=180)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to download {url} (HTTP {response.status_code}).") from exc
            temp_path = local_path.with_suffix(".tif.part")
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            temp_path.replace(local_path)
            with rasterio.open(local_path) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
        else:
            raise

    profile.update(dtype="uint8", nodata=0, compress="deflate", count=1)
    WORLDCOVER_UTM31N_PATH.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(WORLDCOVER_UTM31N_PATH, "w", **profile) as out:
        out.write(dst, 1)
    print(f"[process] Saved WorldCover (EPSG:32631): {WORLDCOVER_UTM31N_PATH}")
    return WORLDCOVER_UTM31N_PATH


def compute_dist_open_water(force: bool = False) -> Path:
    """Euclidean distance in meters from every cell to Copernicus open water (ocean/lagoon)."""
    if DIST_OPEN_WATER_UTM31N_PATH.exists() and not force:
        print(f"[cache] Using existing distance-to-open-water raster: {DIST_OPEN_WATER_UTM31N_PATH}")
        return DIST_OPEN_WATER_UTM31N_PATH

    with rasterio.open(WBM_UTM31N_PATH) as wbm_src, rasterio.open(DEM_UTM31N_PATH) as dem_src:
        wbm = wbm_src.read(1)
        profile = dem_src.profile.copy()

    open_water = (wbm == 1) | (wbm == 2)
    dist_m = distance_transform_edt(~open_water, sampling=(TARGET_RESOLUTION_M, TARGET_RESOLUTION_M))
    dist_arr = dist_m.astype(np.float32)
    profile.update(dtype="float32", nodata=-9999.0, compress="deflate")
    with rasterio.open(DIST_OPEN_WATER_UTM31N_PATH, "w", **profile) as dst:
        dst.write(dist_arr, 1)
    print(f"[process] Saved distance-to-open-water GeoTIFF: {DIST_OPEN_WATER_UTM31N_PATH}")
    return DIST_OPEN_WATER_UTM31N_PATH


def export_lgas(force: bool = False) -> Path:
    """Clip the Lagos LGA polygons to the study bounding box for the web map."""
    if LAGOS_LGAS_WGS84_PATH.exists() and not force:
        print(f"[cache] Using existing LGA overlay: {LAGOS_LGAS_WGS84_PATH}")
        return LAGOS_LGAS_WGS84_PATH
    if not LAGOS_LGAS_PATH.exists():
        raise FileNotFoundError(f"LGA reference file missing: {LAGOS_LGAS_PATH}")

    gdf = gpd.read_file(LAGOS_LGAS_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84_CRS)
    else:
        gdf = gdf.to_crs(WGS84_CRS)
    min_lon, min_lat, max_lon, max_lat = LAGOS_BBOX_WGS84
    clipped = gpd.clip(gdf, box(min_lon, min_lat, max_lon, max_lat))
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    clipped.to_file(LAGOS_LGAS_WGS84_PATH, driver="GeoJSON")
    print(f"[process] Saved clipped LGA GeoJSON ({len(clipped)} features): {LAGOS_LGAS_WGS84_PATH}")
    return LAGOS_LGAS_WGS84_PATH


def render_context_preview(force: bool = False) -> Path:
    if CONTEXT_PNG_PATH.exists() and not force:
        print(f"[cache] Using existing land cover / coastal preview: {CONTEXT_PNG_PATH}")
        return CONTEXT_PNG_PATH

    with rasterio.open(WORLDCOVER_UTM31N_PATH) as cover_src, \
         rasterio.open(DIST_OPEN_WATER_UTM31N_PATH) as dist_src, \
         rasterio.open(HILLSHADE_UTM31N_PATH) as hs_src, \
         rasterio.open(WBM_UTM31N_PATH) as wbm_src:
        cover = cover_src.read(1)
        dist = dist_src.read(1)
        hs = hs_src.read(1)
        wbm = wbm_src.read(1)
        bounds = cover_src.bounds

    extent_km = [bounds.left / 1000.0, bounds.right / 1000.0, bounds.bottom / 1000.0, bounds.top / 1000.0]
    water_mask = np.ma.masked_where(~((wbm == 1) | (wbm == 2)), np.ones_like(wbm))

    # Collapse WorldCover to a few display classes for the diagnostic figure.
    display = np.zeros_like(cover, dtype=np.uint8)
    display[np.isin(cover, [10, 20, 30])] = 1  # vegetation
    display[np.isin(cover, [40])] = 2          # cropland
    display[np.isin(cover, [50])] = 3          # built-up
    display[np.isin(cover, [60])] = 4          # bare
    display[np.isin(cover, [80, 90, 95])] = 5  # water / wetland
    cmap = ListedColormap(["#f5f5f5", "#2ca25f", "#fec44f", "#de2d26", "#d9d9d9", "#3182bd"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), dpi=150, constrained_layout=True)
    ax0 = axes[0]
    ax0.imshow(hs, cmap="gray", extent=extent_km, origin="upper", vmin=40, vmax=245)
    ax0.imshow(display, cmap=cmap, extent=extent_km, origin="upper", vmin=0, vmax=5, alpha=0.72)
    ax0.set_title("ESA WorldCover 2021 (EPSG:32631)", fontsize=12, fontweight="bold")
    ax0.set_xlabel("UTM Zone 31N Easting (km)")
    ax0.set_ylabel("UTM Zone 31N Northing (km)")

    ax1 = axes[1]
    ax1.imshow(hs, cmap="gray", extent=extent_km, origin="upper", vmin=40, vmax=245)
    dist_plot = np.ma.masked_where((wbm == 1) | (wbm == 2), dist)
    im1 = ax1.imshow(dist_plot, cmap="YlOrRd_r", extent=extent_km, origin="upper", vmin=0.0, vmax=4000.0, alpha=0.80)
    ax1.imshow(water_mask, cmap="Blues", extent=extent_km, origin="upper", vmin=0, vmax=1.4, alpha=0.65)
    cbar = fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label("Distance to lagoon / ocean (m)")
    ax1.set_title("Distance to Open Water (tidal / lagoon factor)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("UTM Zone 31N Easting (km)")
    ax1.set_ylabel("UTM Zone 31N Northing (km)")

    fig.savefig(CONTEXT_PNG_PATH)
    plt.close(fig)
    print(f"[process] Saved land cover / coastal preview: {CONTEXT_PNG_PATH}")
    return CONTEXT_PNG_PATH


def print_summary() -> None:
    with rasterio.open(WORLDCOVER_UTM31N_PATH) as cover_src, \
         rasterio.open(DIST_OPEN_WATER_UTM31N_PATH) as dist_src, \
         rasterio.open(WBM_UTM31N_PATH) as wbm_src:
        cover = cover_src.read(1)
        dist = dist_src.read(1)
        wbm = wbm_src.read(1)
    land = ~((wbm == 1) | (wbm == 2))
    built = np.mean((cover == 50) & land) * 100.0
    near = np.mean((dist <= 2000.0) & land) * 100.0
    print("\n--- Milestone 5 Land Cover and Coastal Context ---")
    print(f"Built-up share of land     : {built:.2f}%")
    print(f"Land within 2 km of water  : {near:.2f}%")
    print(f"WorldCover raster          : {WORLDCOVER_UTM31N_PATH}")
    print(f"Open-water distance raster : {DIST_OPEN_WATER_UTM31N_PATH}")
    print(f"LGA overlay                : {LAGOS_LGAS_WGS84_PATH}")


def run_step5(force: bool = False) -> None:
    warp_worldcover(force=force)
    compute_dist_open_water(force=force)
    export_lgas(force=force)
    render_context_preview(force=force)
    print_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build land cover, coastal distance, and LGA layers.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_step5(force=args.force)


if __name__ == "__main__":
    main()
