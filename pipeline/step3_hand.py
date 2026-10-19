"""Milestone 3: Download OSM waterways and compute HAND, distance to nearest drainage, and slope in EPSG:32631."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_erosion, distance_transform_edt
import whitebox

from pipeline.config import (
    DEM_FILLED_UTM31N_PATH,
    DEM_NODATA,
    DEM_UTM31N_PATH,
    DIST_DRAINAGE_UTM31N_PATH,
    DRAINAGE_TARGET_UTM31N_PATH,
    HAND_PNG_PATH,
    HAND_UTM31N_PATH,
    HILLSHADE_UTM31N_PATH,
    LAGOS_BBOX_WGS84,
    OSM_WATERWAYS_TAGS,
    OSM_WATERWAYS_UTM31N_PATH,
    OSM_WATERWAYS_WGS84_PATH,
    SLOPE_UTM31N_PATH,
    STREAMS_RASTER_UTM31N_PATH,
    SYSTEM_CANALS_PATH,
    TARGET_RESOLUTION_M,
    UTM_CRS,
    WBM_UTM31N_PATH,
    WGS84_CRS,
)


def get_wbt() -> whitebox.WhiteboxTools:
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    return wbt


def download_osm_waterways(force: bool = False) -> tuple[Path, Path]:
    """Fetch OpenStreetMap waterways within the Lagos bounding box and reproject to EPSG:32631."""
    if OSM_WATERWAYS_WGS84_PATH.exists() and OSM_WATERWAYS_UTM31N_PATH.exists() and not force:
        print(f"[cache] Using existing OSM waterways: {OSM_WATERWAYS_UTM31N_PATH}")
        return OSM_WATERWAYS_WGS84_PATH, OSM_WATERWAYS_UTM31N_PATH

    print(f"[download] Querying OpenStreetMap Overpass API for waterways in {LAGOS_BBOX_WGS84} ...")
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(OSM_WATERWAYS_WGS84_PATH.parent / "osmnx_cache")
    ox.settings.timeout = 180
    # Query the entire Lagos 1.0 x 0.5 degree box in one Overpass request instead of subdividing.
    ox.settings.max_query_area_size = 25_000_000_000

    gdf = ox.features.features_from_bbox(bbox=LAGOS_BBOX_WGS84, tags=OSM_WATERWAYS_TAGS)
    gdf = gdf.reset_index()

    # Retain line and polygon geometries representing rivers, canals, streams, drains, and ditches.
    valid_geom_types = {"LineString", "MultiLineString", "Polygon", "MultiPolygon"}
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.geom_type.isin(valid_geom_types)].copy()

    keep_cols = [c for c in ("osmid", "id", "waterway", "name", "geometry") if c in gdf.columns]
    gdf = gdf[keep_cols].copy()
    for col in gdf.columns:
        if col != "geometry":
            gdf[col] = gdf[col].astype(str)

    gdf = gdf.set_crs(WGS84_CRS, allow_override=True)
    gdf.to_file(OSM_WATERWAYS_WGS84_PATH, driver="GeoJSON")

    # Reproject to EPSG:32631 before computing lengths or rasterizing onto the metric grid.
    gdf_utm = gdf.to_crs(UTM_CRS)
    gdf_utm["length_m"] = gdf_utm.geometry.length.round(1)
    gdf_utm.to_file(OSM_WATERWAYS_UTM31N_PATH, driver="GeoJSON")

    print(f"[download] Saved {len(gdf_utm):,} OSM waterway features to {OSM_WATERWAYS_UTM31N_PATH}")
    return OSM_WATERWAYS_WGS84_PATH, OSM_WATERWAYS_UTM31N_PATH


def build_drainage_target_raster(force: bool = False) -> Path:
    """Create unified drainage target raster (D8 streams + OSM waterways + shoreline outlets) in EPSG:32631."""
    if DRAINAGE_TARGET_UTM31N_PATH.exists() and not force:
        print(f"[cache] Using existing drainage target raster: {DRAINAGE_TARGET_UTM31N_PATH}")
        return DRAINAGE_TARGET_UTM31N_PATH

    with rasterio.open(STREAMS_RASTER_UTM31N_PATH) as str_src, \
         rasterio.open(WBM_UTM31N_PATH) as wbm_src:
        streams = str_src.read(1)
        str_nodata = str_src.nodata
        wbm = wbm_src.read(1)
        transform = str_src.transform
        profile = str_src.profile.copy()
        height, width = str_src.height, str_src.width

    gdf_osm = gpd.read_file(OSM_WATERWAYS_UTM31N_PATH)
    osm_shapes = [(geom, 1) for geom in gdf_osm.geometry if geom is not None and not geom.is_empty]
    if SYSTEM_CANALS_PATH.exists():
        gdf_canals = gpd.read_file(SYSTEM_CANALS_PATH)
        if gdf_canals.crs is None:
            gdf_canals = gdf_canals.set_crs(WGS84_CRS)
        gdf_canals = gdf_canals.to_crs(UTM_CRS)
        osm_shapes.extend(
            (geom, 1) for geom in gdf_canals.geometry if geom is not None and not geom.is_empty
        )
        print(f"[process] Merged {len(gdf_canals)} System 1–6 canal trunks into drainage target.")
    osm_raster = rasterize(
        shapes=osm_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    open_water = (wbm == 1) | (wbm == 2)
    interior_open_water = binary_erosion(open_water, iterations=1, border_value=1)
    shoreline_ring = open_water & (~interior_open_water)
    copernicus_rivers = wbm == 3

    d8_streams = (streams != str_nodata) & np.isfinite(streams) & (streams > 0)

    # Reason for combining D8 streams, OSM waterways, Copernicus rivers, and the 1-pixel shoreline ring:
    # 1. D8 streams capture topographic valley bottoms where runoff converges.
    # 2. OSM waterways capture engineered urban canals (e.g., System 1-6 canals in mainland Lagos).
    # 3. Including the shoreline ring gives coastal catchments (Victoria Island, Ikoyi, Lekki) a 0m base-level
    #    drainage target when their flow paths enter Lagos Lagoon or the Atlantic directly.
    drainage_mask = (d8_streams | (osm_raster > 0) | copernicus_rivers | shoreline_ring) & (~interior_open_water)

    drainage_arr = np.where(drainage_mask, 1, 0).astype(np.uint8)
    profile.update(dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(DRAINAGE_TARGET_UTM31N_PATH, "w", **profile) as dst:
        dst.write(drainage_arr, 1)

    print(f"[process] Saved unified drainage target raster: {DRAINAGE_TARGET_UTM31N_PATH}")
    return DRAINAGE_TARGET_UTM31N_PATH


def compute_hand_raster(force: bool = False) -> Path:
    """Compute Height Above Nearest Drainage (HAND) in meters for every land cell in EPSG:32631."""
    if HAND_UTM31N_PATH.exists() and not force:
        print(f"[cache] Using existing HAND raster: {HAND_UTM31N_PATH}")
        return HAND_UTM31N_PATH

    wbt = get_wbt()
    ok = wbt.elevation_above_stream(
        dem=str(DEM_FILLED_UTM31N_PATH),
        streams=str(DRAINAGE_TARGET_UTM31N_PATH),
        output=str(HAND_UTM31N_PATH),
    )
    if ok != 0 or not HAND_UTM31N_PATH.exists():
        raise RuntimeError("WhiteboxTools elevation_above_stream failed.")

    # Post-process edge catchments that drain across the outer study-area bounding box border
    # before intersecting an interior stream channel.
    with rasterio.open(HAND_UTM31N_PATH) as hand_src, \
         rasterio.open(DEM_FILLED_UTM31N_PATH) as fill_src, \
         rasterio.open(DRAINAGE_TARGET_UTM31N_PATH) as drn_src, \
         rasterio.open(WBM_UTM31N_PATH) as wbm_src:
        hand = hand_src.read(1)
        hand_nodata = hand_src.nodata
        dem_filled = fill_src.read(1)
        fill_nodata = fill_src.nodata
        drn = drn_src.read(1)
        wbm = wbm_src.read(1)
        profile = fill_src.profile.copy()

    open_water = (wbm == 1) | (wbm == 2)
    valid_land = (~open_water) & (dem_filled != fill_nodata) & np.isfinite(dem_filled)
    valid_hand = (hand != hand_nodata) & np.isfinite(hand) & (hand >= 0.0)

    # Reason for nearest-drainage fallback on boundary-clipped cells:
    # Catchments truncated by the north/east/west bounding box edge exit the grid before reaching
    # a 1,000-cell channel; referencing the nearest drainage cell elevation prevents artificial NoData gaps on land.
    missing_land = valid_land & (~valid_hand)
    if np.any(missing_land):
        _, (nearest_r, nearest_c) = distance_transform_edt(drn == 0, return_indices=True)
        fallback_hand = np.maximum(0.0, dem_filled - dem_filled[nearest_r, nearest_c])
        hand[missing_land] = fallback_hand[missing_land]

    hand = np.where(valid_land, np.maximum(0.0, hand), DEM_NODATA).astype(np.float32)

    profile.update(dtype="float32", nodata=DEM_NODATA, compress="deflate")
    with rasterio.open(HAND_UTM31N_PATH, "w", **profile) as dst:
        dst.write(hand, 1)

    print(f"[process] Saved HAND GeoTIFF: {HAND_UTM31N_PATH}")
    return HAND_UTM31N_PATH


def compute_distance_to_drainage(force: bool = False) -> Path:
    """Compute Euclidean distance (in meters, EPSG:32631) to the nearest D8 stream or OSM waterway."""
    if DIST_DRAINAGE_UTM31N_PATH.exists() and not force:
        print(f"[cache] Using existing distance-to-drainage raster: {DIST_DRAINAGE_UTM31N_PATH}")
        return DIST_DRAINAGE_UTM31N_PATH

    with rasterio.open(DRAINAGE_TARGET_UTM31N_PATH) as drn_src, \
         rasterio.open(WBM_UTM31N_PATH) as wbm_src, \
         rasterio.open(DEM_UTM31N_PATH) as dem_src:
        drn = drn_src.read(1)
        wbm = wbm_src.read(1)
        dem = dem_src.read(1)
        profile = dem_src.profile.copy()

    open_water = (wbm == 1) | (wbm == 2)
    valid_land = (~open_water) & (dem != dem_src.nodata) & np.isfinite(dem)

    # All distances are computed strictly in EPSG:32631 where each cell is 30.0m x 30.0m.
    water_or_drainage = (drn > 0) | open_water
    dist_m = distance_transform_edt(~water_or_drainage, sampling=(TARGET_RESOLUTION_M, TARGET_RESOLUTION_M))
    dist_arr = np.where(valid_land, dist_m, DEM_NODATA).astype(np.float32)

    profile.update(dtype="float32", nodata=DEM_NODATA, compress="deflate")
    with rasterio.open(DIST_DRAINAGE_UTM31N_PATH, "w", **profile) as dst:
        dst.write(dist_arr, 1)

    print(f"[process] Saved distance-to-drainage GeoTIFF: {DIST_DRAINAGE_UTM31N_PATH}")
    return DIST_DRAINAGE_UTM31N_PATH


def compute_slope_degrees(force: bool = False) -> Path:
    """Compute terrain slope in degrees on the EPSG:32631 DEM."""
    if SLOPE_UTM31N_PATH.exists() and not force:
        print(f"[cache] Using existing slope raster: {SLOPE_UTM31N_PATH}")
        return SLOPE_UTM31N_PATH

    wbt = get_wbt()
    ok = wbt.slope(
        dem=str(DEM_UTM31N_PATH),
        output=str(SLOPE_UTM31N_PATH),
        units="degrees",
    )
    if ok != 0 or not SLOPE_UTM31N_PATH.exists():
        raise RuntimeError("WhiteboxTools slope failed.")

    with rasterio.open(SLOPE_UTM31N_PATH) as slp_src, \
         rasterio.open(WBM_UTM31N_PATH) as wbm_src, \
         rasterio.open(DEM_UTM31N_PATH) as dem_src:
        slope = slp_src.read(1)
        slp_nodata = slp_src.nodata
        wbm = wbm_src.read(1)
        profile = dem_src.profile.copy()

    open_water = (wbm == 1) | (wbm == 2)
    valid_land = (~open_water) & (slope != slp_nodata) & np.isfinite(slope)
    slope_clean = np.where(valid_land, np.clip(slope, 0.0, 90.0), DEM_NODATA).astype(np.float32)

    profile.update(dtype="float32", nodata=DEM_NODATA, compress="deflate")
    with rasterio.open(SLOPE_UTM31N_PATH, "w", **profile) as dst:
        dst.write(slope_clean, 1)

    print(f"[process] Saved slope (degrees) GeoTIFF: {SLOPE_UTM31N_PATH}")
    return SLOPE_UTM31N_PATH


def render_hand_preview(force: bool = False) -> Path:
    """Render a 3-panel diagnostic figure showing HAND, Distance to Drainage, and Slope."""
    if HAND_PNG_PATH.exists() and not force:
        print(f"[cache] Using existing HAND preview PNG: {HAND_PNG_PATH}")
        return HAND_PNG_PATH

    with rasterio.open(HAND_UTM31N_PATH) as hand_src, \
         rasterio.open(DIST_DRAINAGE_UTM31N_PATH) as dist_src, \
         rasterio.open(SLOPE_UTM31N_PATH) as slp_src, \
         rasterio.open(HILLSHADE_UTM31N_PATH) as hs_src, \
         rasterio.open(WBM_UTM31N_PATH) as wbm_src:
        hand = hand_src.read(1)
        dist = dist_src.read(1)
        slope = slp_src.read(1)
        hs = hs_src.read(1)
        wbm = wbm_src.read(1)
        bounds = hand_src.bounds

    extent_km = [
        bounds.left / 1000.0,
        bounds.right / 1000.0,
        bounds.bottom / 1000.0,
        bounds.top / 1000.0,
    ]
    water_mask = np.ma.masked_where(~((wbm == 1) | (wbm == 2)), np.ones_like(wbm))

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), dpi=150, constrained_layout=True)

    # Panel 1: HAND
    ax0 = axes[0]
    ax0.imshow(hs, cmap="gray", extent=extent_km, origin="upper", vmin=40, vmax=245)
    hand_plot = np.ma.masked_where((hand == DEM_NODATA) | ~np.isfinite(hand), hand)
    im0 = ax0.imshow(hand_plot, cmap="RdYlBu_r", extent=extent_km, origin="upper", vmin=0.0, vmax=20.0, alpha=0.80)
    ax0.imshow(water_mask, cmap="Blues", extent=extent_km, origin="upper", vmin=0, vmax=1.4, alpha=0.65)
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.035, pad=0.02)
    cbar0.set_label("HAND (m, clipped at 20m)")
    ax0.set_title("Height Above Nearest Drainage (HAND)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("UTM 31N Easting (km)")
    ax0.set_ylabel("UTM 31N Northing (km)")

    # Panel 2: Distance to Nearest Drainage / OSM Waterway
    ax1 = axes[1]
    ax1.imshow(hs, cmap="gray", extent=extent_km, origin="upper", vmin=40, vmax=245)
    dist_plot = np.ma.masked_where((dist == DEM_NODATA) | ~np.isfinite(dist), dist)
    im1 = ax1.imshow(dist_plot, cmap="viridis_r", extent=extent_km, origin="upper", vmin=0.0, vmax=1500.0, alpha=0.80)
    ax1.imshow(water_mask, cmap="Blues", extent=extent_km, origin="upper", vmin=0, vmax=1.4, alpha=0.65)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.02)
    cbar1.set_label("Distance to Drainage (m, clipped at 1500m)")
    ax1.set_title("Distance to Nearest Drainage / OSM Waterway", fontsize=11, fontweight="bold")
    ax1.set_xlabel("UTM 31N Easting (km)")
    ax1.set_ylabel("UTM 31N Northing (km)")

    # Panel 3: Slope (degrees)
    ax2 = axes[2]
    ax2.imshow(hs, cmap="gray", extent=extent_km, origin="upper", vmin=40, vmax=245)
    slp_plot = np.ma.masked_where((slope == DEM_NODATA) | ~np.isfinite(slope), slope)
    im2 = ax2.imshow(slp_plot, cmap="magma_r", extent=extent_km, origin="upper", vmin=0.0, vmax=8.0, alpha=0.80)
    ax2.imshow(water_mask, cmap="Blues", extent=extent_km, origin="upper", vmin=0, vmax=1.4, alpha=0.65)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.02)
    cbar2.set_label("Slope (degrees, clipped at 8°)")
    ax2.set_title("Terrain Slope (EPSG:32631)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("UTM 31N Easting (km)")
    ax2.set_ylabel("UTM 31N Northing (km)")

    fig.savefig(HAND_PNG_PATH)
    plt.close(fig)

    print(f"[process] Saved HAND, distance, and slope preview PNG: {HAND_PNG_PATH}")
    return HAND_PNG_PATH


def print_summary() -> None:
    """Print summary statistics for HAND, distance to drainage, and slope."""
    with rasterio.open(HAND_UTM31N_PATH) as hand_src, \
         rasterio.open(DIST_DRAINAGE_UTM31N_PATH) as dist_src, \
         rasterio.open(SLOPE_UTM31N_PATH) as slp_src:
        hand = hand_src.read(1)
        dist = dist_src.read(1)
        slope = slp_src.read(1)

        v_hand = hand[(hand != DEM_NODATA) & np.isfinite(hand)]
        v_dist = dist[(dist != DEM_NODATA) & np.isfinite(dist)]
        v_slp = slope[(slope != DEM_NODATA) & np.isfinite(slope)]

    gdf_osm = gpd.read_file(OSM_WATERWAYS_UTM31N_PATH)
    osm_km = gdf_osm["length_m"].sum() / 1000.0

    print("\n--- Milestone 3 HAND, Distance, and Slope Summary ---")
    print(f"OSM waterway features     : {len(gdf_osm):,} ({osm_km:.1f} km total length in EPSG:32631)")
    print(f"Valid land cells modeled  : {v_hand.size:,}")
    print(
        f"HAND (m)                  : min={v_hand.min():.2f}, p25={np.percentile(v_hand, 25):.2f}, "
        f"median={np.median(v_hand):.2f}, p75={np.percentile(v_hand, 75):.2f}, max={v_hand.max():.2f}"
    )
    print(f"Land with HAND <= 2.0 m   : {100.0 * np.mean(v_hand <= 2.0):.2f}%")
    print(f"Land with HAND <= 5.0 m   : {100.0 * np.mean(v_hand <= 5.0):.2f}%")
    print(
        f"Distance to drainage (m)  : min={v_dist.min():.1f}, median={np.median(v_dist):.1f}, "
        f"p90={np.percentile(v_dist, 90):.1f}, max={v_dist.max():.1f}"
    )
    print(
        f"Slope (degrees)           : min={v_slp.min():.2f}, median={np.median(v_slp):.2f}, "
        f"p90={np.percentile(v_slp, 90):.2f}, max={v_slp.max():.2f}"
    )
    print(f"HAND raster               : {HAND_UTM31N_PATH}")
    print(f"Distance raster           : {DIST_DRAINAGE_UTM31N_PATH}")
    print(f"Slope raster              : {SLOPE_UTM31N_PATH}")


def run_step3(force: bool = False) -> None:
    download_osm_waterways(force=force)
    build_drainage_target_raster(force=force)
    compute_hand_raster(force=force)
    compute_distance_to_drainage(force=force)
    compute_slope_degrees(force=force)
    render_hand_preview(force=force)
    print_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute HAND, distance to drainage, and slope.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_step3(force=args.force)
