"""Milestone 4: Classify flood susceptibility into Low/Medium/High classes and export raster, vector, and PNG overlays."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from shapely.geometry import shape

from pipeline.config import (
    COASTAL_HIGH_MAX_M,
    COASTAL_INFLUENCE_M,
    COASTAL_MEDIUM_MAX_M,
    DEM_NODATA,
    DEM_UTM31N_PATH,
    DIST_DRAINAGE_UTM31N_PATH,
    DIST_HIGH_MAX_M,
    DIST_MEDIUM_MAX_M,
    DIST_OPEN_WATER_UTM31N_PATH,
    HAND_HIGH_MAX_M,
    HAND_MEDIUM_MAX_M,
    HAND_UTM31N_PATH,
    HILLSHADE_UTM31N_PATH,
    LAGOS_BBOX_WGS84,
    RISK_CLASS_COLORS_RGBA,
    RISK_CLASS_HIGH,
    RISK_CLASS_LABELS,
    RISK_CLASS_LOW,
    RISK_CLASS_MEDIUM,
    RISK_CLASS_NODATA,
    RISK_MAP_PNG_PATH,
    RISK_METADATA_JSON_PATH,
    RISK_OVERLAY_HIGH_PNG_PATH,
    RISK_OVERLAY_LOW_PNG_PATH,
    RISK_OVERLAY_MEDIUM_PNG_PATH,
    RISK_OVERLAY_PNG_PATH,
    RISK_SCORE_HIGH_MIN,
    RISK_SCORE_MEDIUM_MIN,
    RISK_SCORE_UTM31N_PATH,
    RISK_UTM31N_PATH,
    RISK_WGS84_PATH,
    RISK_ZONES_GEOJSON_PATH,
    SCORE_OVERLAY_PNG_PATH,
    SLOPE_FLAT_MAX_DEG,
    SLOPE_MODERATE_MAX_DEG,
    SLOPE_UTM31N_PATH,
    STREAMS_GEOJSON_UTM31N_PATH,
    UTM_CRS,
    WBM_UTM31N_PATH,
    WEIGHT_DIST,
    WEIGHT_HAND,
    WEIGHT_IMPERVIOUS,
    WEIGHT_SLOPE,
    WGS84_CRS,
    WORLDCOVER_UTM31N_PATH,
)
from pipeline.scoring import compute_risk_rasters


def classify_rasters(force: bool = False) -> tuple[Path, Path]:
    """Generate classified GeoTIFF in EPSG:32631 and warp to EPSG:4326 for web overlay alignment."""
    if RISK_UTM31N_PATH.exists() and RISK_WGS84_PATH.exists() and RISK_SCORE_UTM31N_PATH.exists() and not force:
        print(f"[cache] Using existing classified risk rasters: {RISK_UTM31N_PATH}")
        return RISK_UTM31N_PATH, RISK_WGS84_PATH

    with rasterio.open(HAND_UTM31N_PATH) as hand_src, \
         rasterio.open(DIST_DRAINAGE_UTM31N_PATH) as dist_src, \
         rasterio.open(SLOPE_UTM31N_PATH) as slp_src, \
         rasterio.open(WBM_UTM31N_PATH) as wbm_src, \
         rasterio.open(WORLDCOVER_UTM31N_PATH) as cover_src, \
         rasterio.open(DEM_UTM31N_PATH) as dem_src, \
         rasterio.open(DIST_OPEN_WATER_UTM31N_PATH) as dow_src:
        hand = hand_src.read(1)
        dist = dist_src.read(1)
        slope = slp_src.read(1)
        wbm = wbm_src.read(1)
        worldcover = cover_src.read(1)
        elevation = dem_src.read(1)
        dist_open_water = dow_src.read(1)
        utm_profile = hand_src.profile.copy()
        utm_transform = hand_src.transform
        utm_crs = hand_src.crs
        height, width = hand_src.height, hand_src.width

    open_water = (wbm == 1) | (wbm == 2)
    valid_land = (
        (~open_water)
        & (hand != DEM_NODATA)
        & (dist != DEM_NODATA)
        & (slope != DEM_NODATA)
        & np.isfinite(hand)
        & np.isfinite(dist)
        & np.isfinite(slope)
    )

    risk_utm, score_utm, _factors = compute_risk_rasters(
        hand, dist, slope, worldcover, elevation, dist_open_water, valid_land
    )

    utm_profile.update(dtype="uint8", nodata=RISK_CLASS_NODATA, compress="deflate")
    with rasterio.open(RISK_UTM31N_PATH, "w", **utm_profile) as dst:
        dst.write(risk_utm, 1)
    print(f"[process] Saved classified risk GeoTIFF (EPSG:32631): {RISK_UTM31N_PATH}")

    score_profile = utm_profile.copy()
    score_profile.update(dtype="float32", nodata=DEM_NODATA, compress="deflate")
    score_out = np.where(np.isfinite(score_utm), score_utm, DEM_NODATA).astype(np.float32)
    with rasterio.open(RISK_SCORE_UTM31N_PATH, "w", **score_profile) as dst:
        dst.write(score_out, 1)
    print(f"[process] Saved continuous score GeoTIFF (EPSG:32631): {RISK_SCORE_UTM31N_PATH}")

    # Warp the classified raster to EPSG:4326 using nearest-neighbor resampling.
    # Reason: Leaflet L.imageOverlay expects a lat/lon aligned image; nearest-neighbor
    # preserves exact integer class IDs (0, 1, 2, 3) without blending class values at boundaries.
    min_lon, min_lat, max_lon, max_lat = LAGOS_BBOX_WGS84
    wgs84_width = width
    wgs84_height = height
    wgs84_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, wgs84_width, wgs84_height)
    risk_wgs84 = np.zeros((wgs84_height, wgs84_width), dtype=np.uint8)

    reproject(
        source=risk_utm,
        destination=risk_wgs84,
        src_transform=utm_transform,
        src_crs=utm_crs,
        src_nodata=RISK_CLASS_NODATA,
        dst_transform=wgs84_transform,
        dst_crs=CRS.from_string(WGS84_CRS),
        dst_nodata=RISK_CLASS_NODATA,
        resampling=Resampling.nearest,
    )

    wgs84_profile = {
        "driver": "GTiff",
        "height": wgs84_height,
        "width": wgs84_width,
        "count": 1,
        "dtype": "uint8",
        "crs": CRS.from_string(WGS84_CRS),
        "transform": wgs84_transform,
        "nodata": RISK_CLASS_NODATA,
        "compress": "deflate",
    }
    with rasterio.open(RISK_WGS84_PATH, "w", **wgs84_profile) as dst:
        dst.write(risk_wgs84, 1)
    print(f"[process] Saved classified risk GeoTIFF (EPSG:4326): {RISK_WGS84_PATH}")

    return RISK_UTM31N_PATH, RISK_WGS84_PATH


def export_web_overlay_and_vectors(force: bool = False) -> tuple[Path, Path, Path]:
    """Export transparent RGBA PNG overlay, generalized GeoJSON vector polygons, and metadata JSON."""
    if (
        RISK_OVERLAY_PNG_PATH.exists()
        and RISK_OVERLAY_HIGH_PNG_PATH.exists()
        and SCORE_OVERLAY_PNG_PATH.exists()
        and RISK_ZONES_GEOJSON_PATH.exists()
        and RISK_METADATA_JSON_PATH.exists()
        and not force
    ):
        print(f"[cache] Using existing web overlay PNG and GeoJSON: {RISK_OVERLAY_PNG_PATH}")
        return RISK_OVERLAY_PNG_PATH, RISK_ZONES_GEOJSON_PATH, RISK_METADATA_JSON_PATH

    with rasterio.open(RISK_WGS84_PATH) as src:
        risk_wgs84 = src.read(1)
        bounds = src.bounds
        transform = src.transform

    # Build RGBA array for the transparent web overlay PNG.
    rgba = np.zeros((risk_wgs84.shape[0], risk_wgs84.shape[1], 4), dtype=np.uint8)
    for class_id, color_tuple in RISK_CLASS_COLORS_RGBA.items():
        mask = risk_wgs84 == class_id
        rgba[mask] = color_tuple

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(RISK_OVERLAY_PNG_PATH, format="PNG", optimize=True)
    print(f"[process] Saved transparent RGBA web map overlay: {RISK_OVERLAY_PNG_PATH}")

    class_overlay_paths = {
        RISK_CLASS_HIGH: RISK_OVERLAY_HIGH_PNG_PATH,
        RISK_CLASS_MEDIUM: RISK_OVERLAY_MEDIUM_PNG_PATH,
        RISK_CLASS_LOW: RISK_OVERLAY_LOW_PNG_PATH,
    }
    for class_id, path in class_overlay_paths.items():
        class_rgba = np.zeros_like(rgba)
        mask = risk_wgs84 == class_id
        class_rgba[mask] = RISK_CLASS_COLORS_RGBA[class_id]
        Image.fromarray(class_rgba, mode="RGBA").save(path, format="PNG", optimize=True)
        print(f"[process] Saved class overlay: {path}")

    with rasterio.open(RISK_SCORE_UTM31N_PATH) as score_src:
        score_utm = score_src.read(1)
        score_nodata = score_src.nodata
        score_wgs84 = np.full(risk_wgs84.shape, np.nan, dtype=np.float32)
        reproject(
            source=score_utm,
            destination=score_wgs84,
            src_transform=score_src.transform,
            src_crs=score_src.crs,
            src_nodata=score_nodata,
            dst_transform=transform,
            dst_crs=CRS.from_string(WGS84_CRS),
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    cmap = plt.get_cmap("RdYlGn_r")
    finite = np.isfinite(score_wgs84) & (risk_wgs84 > 0)
    norm = np.clip((score_wgs84 - 1.0) / 2.0, 0.0, 1.0)
    score_rgba = np.zeros((risk_wgs84.shape[0], risk_wgs84.shape[1], 4), dtype=np.uint8)
    colored = (np.asarray(cmap(norm)) * 255.0).astype(np.uint8)
    score_rgba[finite] = colored[finite]
    score_rgba[finite, 3] = 185
    score_img = Image.fromarray(score_rgba, mode="RGBA")
    score_img = score_img.resize((max(1, score_img.width // 2), max(1, score_img.height // 2)), Image.BILINEAR)
    score_img.save(SCORE_OVERLAY_PNG_PATH, format="PNG", optimize=True)
    print(f"[process] Saved continuous score overlay: {SCORE_OVERLAY_PNG_PATH}")

    # Export a generalized vector polygon GeoJSON (downsampled 8x using nearest-neighbor)
    # so vector inspection tools can load risk zones without a 500MB polygon file.
    step = 8
    coarse_risk = risk_wgs84[::step, ::step]
    coarse_transform = transform * transform.scale(step, step)
    records = []
    for geom_dict, val in shapes(coarse_risk, mask=(coarse_risk > 0), transform=coarse_transform):
        cls_id = int(val)
        records.append(
            {
                "risk_class": cls_id,
                "risk_label": RISK_CLASS_LABELS.get(cls_id, "Unknown"),
                "geometry": shape(geom_dict),
            }
        )
    gdf_zones = gpd.GeoDataFrame(records, crs=WGS84_CRS)
    gdf_zones = gdf_zones.dissolve(by=["risk_class", "risk_label"], as_index=False)
    gdf_zones.to_file(RISK_ZONES_GEOJSON_PATH, driver="GeoJSON")
    print(f"[process] Saved dissolved vector risk zones GeoJSON: {RISK_ZONES_GEOJSON_PATH}")

    # Compute exact class shares from the EPSG:32631 metric raster.
    with rasterio.open(RISK_UTM31N_PATH) as utm_src:
        risk_utm = utm_src.read(1)

    land_cells = int(np.sum(risk_utm > 0))
    low_cells = int(np.sum(risk_utm == RISK_CLASS_LOW))
    med_cells = int(np.sum(risk_utm == RISK_CLASS_MEDIUM))
    high_cells = int(np.sum(risk_utm == RISK_CLASS_HIGH))
    water_cells = int(np.sum(risk_utm == RISK_CLASS_NODATA))

    metadata = {
        "crs_analysis": UTM_CRS,
        "crs_web_overlay": WGS84_CRS,
        "leaflet_bounds": [
            [float(bounds.bottom), float(bounds.left)],
            [float(bounds.top), float(bounds.right)],
        ],
        "thresholds": {
            "hand_high_max_m": HAND_HIGH_MAX_M,
            "hand_medium_max_m": HAND_MEDIUM_MAX_M,
            "dist_high_max_m": DIST_HIGH_MAX_M,
            "dist_medium_max_m": DIST_MEDIUM_MAX_M,
            "slope_flat_max_deg": SLOPE_FLAT_MAX_DEG,
            "slope_moderate_max_deg": SLOPE_MODERATE_MAX_DEG,
            "coastal_high_max_m": COASTAL_HIGH_MAX_M,
            "coastal_medium_max_m": COASTAL_MEDIUM_MAX_M,
            "coastal_influence_m": COASTAL_INFLUENCE_M,
            "weights": {
                "hand": WEIGHT_HAND,
                "distance_to_drainage": WEIGHT_DIST,
                "slope": WEIGHT_SLOPE,
                "impervious": WEIGHT_IMPERVIOUS,
            },
            "score_cutoffs": {
                "high_min": RISK_SCORE_HIGH_MIN,
                "medium_min": RISK_SCORE_MEDIUM_MIN,
            },
            "combine": "max(runoff_weighted, coastal) within 2 km of open water",
        },
        "overlays": {
            "classified": "/static/risk_overlay_wgs84.png",
            "high": "/static/risk_overlay_high_wgs84.png",
            "medium": "/static/risk_overlay_medium_wgs84.png",
            "low": "/static/risk_overlay_low_wgs84.png",
            "score": "/static/score_overlay_wgs84.png",
            "lgas": "/static/lagos_lgas_wgs84.geojson",
            "streams": "/static/streams_wgs84.geojson",
        },
        "classes": {
            "1": {
                "label": "Low",
                "color_hex": "#2ca25f",
                "land_cells": low_cells,
                "land_area_km2": round(low_cells * 900.0 / 1e6, 2),
                "land_share_pct": round(100.0 * low_cells / land_cells, 2),
            },
            "2": {
                "label": "Medium",
                "color_hex": "#feb24c",
                "land_cells": med_cells,
                "land_area_km2": round(med_cells * 900.0 / 1e6, 2),
                "land_share_pct": round(100.0 * med_cells / land_cells, 2),
            },
            "3": {
                "label": "High",
                "color_hex": "#e31a1c",
                "land_cells": high_cells,
                "land_area_km2": round(high_cells * 900.0 / 1e6, 2),
                "land_share_pct": round(100.0 * high_cells / land_cells, 2),
            },
            "0": {
                "label": "Water / NoData",
                "cells": water_cells,
                "area_km2": round(water_cells * 900.0 / 1e6, 2),
            },
        },
    }
    with open(RISK_METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[process] Saved overlay metadata JSON: {RISK_METADATA_JSON_PATH}")

    return RISK_OVERLAY_PNG_PATH, RISK_ZONES_GEOJSON_PATH, RISK_METADATA_JSON_PATH


def render_risk_map_figure(force: bool = False) -> Path:
    """Render a diagnostic publication map of the Lagos Flood Susceptibility classification."""
    if RISK_MAP_PNG_PATH.exists() and not force:
        print(f"[cache] Using existing risk map figure: {RISK_MAP_PNG_PATH}")
        return RISK_MAP_PNG_PATH

    with rasterio.open(RISK_UTM31N_PATH) as risk_src, \
         rasterio.open(HILLSHADE_UTM31N_PATH) as hs_src, \
         rasterio.open(WBM_UTM31N_PATH) as wbm_src:
        risk = risk_src.read(1)
        hs = hs_src.read(1)
        wbm = wbm_src.read(1)
        bounds = risk_src.bounds

    with open(RISK_METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    gdf_streams = gpd.read_file(STREAMS_GEOJSON_UTM31N_PATH)

    extent_km = [
        bounds.left / 1000.0,
        bounds.right / 1000.0,
        bounds.bottom / 1000.0,
        bounds.top / 1000.0,
    ]

    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=150, constrained_layout=True)
    ax.imshow(hs, cmap="gray", extent=extent_km, origin="upper", vmin=40, vmax=245)

    cmap = ListedColormap(["#2ca25f", "#feb24c", "#e31a1c"])
    risk_plot = np.ma.masked_where(risk == 0, risk)
    ax.imshow(risk_plot, cmap=cmap, extent=extent_km, origin="upper", vmin=1, vmax=3, alpha=0.74)

    water_mask = np.ma.masked_where(~((wbm == 1) | (wbm == 2)), np.ones_like(wbm))
    ax.imshow(water_mask, cmap="Blues", extent=extent_km, origin="upper", vmin=0, vmax=1.4, alpha=0.70)

    gdf_km = gdf_streams.copy()
    gdf_km["geometry"] = gdf_km.geometry.scale(xfact=0.001, yfact=0.001, origin=(0, 0))
    gdf_km.plot(ax=ax, color="#08306b", linewidth=0.45, alpha=0.55)

    ax.set_xlim(extent_km[0], extent_km[1])
    ax.set_ylim(extent_km[2], extent_km[3])
    ax.set_title(
        "Lagos Terrain Flood Susceptibility Map (EPSG:32631, 30m Grid)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("UTM Zone 31N Easting (km)")
    ax.set_ylabel("UTM Zone 31N Northing (km)")

    c_low = meta["classes"]["1"]
    c_med = meta["classes"]["2"]
    c_high = meta["classes"]["3"]

    legend_patches = [
        mpatches.Patch(
            color="#e31a1c",
            label=f"High Susceptibility ({c_high['land_share_pct']}% | {c_high['land_area_km2']} km²)",
        ),
        mpatches.Patch(
            color="#feb24c",
            label=f"Medium Susceptibility ({c_med['land_share_pct']}% | {c_med['land_area_km2']} km²)",
        ),
        mpatches.Patch(
            color="#2ca25f",
            label=f"Low Susceptibility ({c_low['land_share_pct']}% | {c_low['land_area_km2']} km²)",
        ),
        mpatches.Patch(color="#4292c6", label="Permanent Water (Lagoon / Atlantic Ocean)"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", framealpha=0.92, fontsize=9.5)

    fig.savefig(RISK_MAP_PNG_PATH)
    plt.close(fig)
    print(f"[process] Saved risk classification figure: {RISK_MAP_PNG_PATH}")
    return RISK_MAP_PNG_PATH


def print_summary() -> None:
    with open(RISK_METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    c1 = meta["classes"]["1"]
    c2 = meta["classes"]["2"]
    c3 = meta["classes"]["3"]
    c0 = meta["classes"]["0"]

    print("\n--- Milestone 4 Risk Classification Summary ---")
    print(
        f"Thresholds     : HAND High <= {HAND_HIGH_MAX_M}m, Medium <= {HAND_MEDIUM_MAX_M}m | "
        f"Dist High <= {DIST_HIGH_MAX_M:.0f}m, Medium <= {DIST_MEDIUM_MAX_M:.0f}m | "
        f"Slope Flat <= {SLOPE_FLAT_MAX_DEG}°, Mod <= {SLOPE_MODERATE_MAX_DEG}°"
    )
    print(
        f"Weights        : HAND={WEIGHT_HAND:.2f}, Distance={WEIGHT_DIST:.2f}, "
        f"Slope={WEIGHT_SLOPE:.2f}, Impervious={WEIGHT_IMPERVIOUS:.2f} "
        f"(coastal via max() within {COASTAL_INFLUENCE_M:.0f} m)"
    )
    print(f"High (3)       : {c3['land_cells']:,} cells ({c3['land_area_km2']} km², {c3['land_share_pct']}% of land)")
    print(f"Medium (2)     : {c2['land_cells']:,} cells ({c2['land_area_km2']} km², {c2['land_share_pct']}% of land)")
    print(f"Low (1)        : {c1['land_cells']:,} cells ({c1['land_area_km2']} km², {c1['land_share_pct']}% of land)")
    print(f"Water/NoData(0): {c0['cells']:,} cells ({c0['area_km2']} km²)")
    print(f"UTM Raster     : {RISK_UTM31N_PATH}")
    print(f"WGS84 Raster   : {RISK_WGS84_PATH}")
    print(f"Web PNG Overlay: {RISK_OVERLAY_PNG_PATH}")
    print(f"Vector GeoJSON : {RISK_ZONES_GEOJSON_PATH}")
    print(f"Diagnostic Map : {RISK_MAP_PNG_PATH}")


def run_step4(force: bool = False) -> None:
    classify_rasters(force=force)
    export_web_overlay_and_vectors(force=force)
    render_risk_map_figure(force=force)
    print_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify flood susceptibility and export overlays.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_step4(force=args.force)
