"""Milestone 2: Condition DEM, fill depressions, compute D8 flow direction and accumulation, and extract drainage network."""
from __future__ import annotations

from pathlib import Path
import tempfile

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject
import requests
from scipy.ndimage import binary_erosion
import whitebox

from pipeline.config import (
    COPERNICUS_S3_BASE_URL,
    COPERNICUS_TILES,
    D8_ACCUM_UTM31N_PATH,
    D8_POINTER_UTM31N_PATH,
    DEM_FILLED_UTM31N_PATH,
    DEM_HYDRO_INPUT_UTM31N_PATH,
    DEM_NODATA,
    DEM_UTM31N_PATH,
    FLOW_ACCUM_THRESHOLD_CELLS,
    HILLSHADE_UTM31N_PATH,
    HYDROLOGY_PNG_PATH,
    LAGOS_BBOX_WGS84,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    STREAMS_GEOJSON_UTM31N_PATH,
    STREAMS_GEOJSON_WGS84_PATH,
    STREAMS_RASTER_UTM31N_PATH,
    UTM_CRS,
    WBM_UTM31N_PATH,
    WGS84_CRS,
)


def get_wbt() -> whitebox.WhiteboxTools:
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    return wbt


def download_and_reproject_wbm(force: bool = False) -> Path:
    """Download Copernicus Water Body Mask (WBM) tiles and warp to the exact EPSG:32631 DEM grid."""
    if WBM_UTM31N_PATH.exists() and not force:
        print(f"[cache] Using existing UTM 31N Water Body Mask: {WBM_UTM31N_PATH}")
        return WBM_UTM31N_PATH

    wbm_raw_paths: list[Path] = []
    for tile_name in COPERNICUS_TILES:
        wbm_name = tile_name.replace("_DEM", "_WBM")
        filename = f"{wbm_name}.tif"
        local_path = RAW_DIR / filename
        url = f"{COPERNICUS_S3_BASE_URL}/{tile_name}/AUXFILES/{filename}"

        if local_path.exists() and local_path.stat().st_size > 0 and not force:
            print(f"[cache] Using existing WBM tile: {local_path}")
            wbm_raw_paths.append(local_path)
            continue

        print(f"[download] Fetching {url} ...")
        response = requests.get(url, stream=True, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download {url} (HTTP {response.status_code}).")
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"[download] Saved {local_path} ({local_path.stat().st_size / 1e3:.1f} KB)")
        wbm_raw_paths.append(local_path)

    min_lon, min_lat, max_lon, max_lat = LAGOS_BBOX_WGS84
    padded_bbox = (max(3.0, min_lon - 0.02), min_lat - 0.02, max_lon + 0.02, max_lat + 0.02)

    datasets = [rasterio.open(p) for p in wbm_raw_paths]
    try:
        wbm_mosaic, wbm_transform = merge(datasets, bounds=padded_bbox, nodata=0)
    finally:
        for ds in datasets:
            ds.close()

    with rasterio.open(DEM_UTM31N_PATH) as dem_src:
        dst_profile = dem_src.profile.copy()
        dst_height = dem_src.height
        dst_width = dem_src.width
        dst_transform = dem_src.transform
        dst_crs = dem_src.crs

    dst_wbm = np.zeros((dst_height, dst_width), dtype=np.uint8)

    # Reason for nearest-neighbor resampling: WBM stores discrete classes
    # (0=Land, 1=Ocean, 2=Lake/Lagoon, 3=River); interpolation would create invalid class codes.
    reproject(
        source=wbm_mosaic[0].astype(np.uint8),
        destination=dst_wbm,
        src_transform=wbm_transform,
        src_crs=CRS.from_string(WGS84_CRS),
        src_nodata=255,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=255,
        resampling=Resampling.nearest,
    )

    dst_profile.update(dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(WBM_UTM31N_PATH, "w", **dst_profile) as dst:
        dst.write(dst_wbm, 1)

    print(f"[process] Saved EPSG:32631 Water Body Mask: {WBM_UTM31N_PATH}")
    return WBM_UTM31N_PATH


def prepare_hydro_dem(force: bool = False) -> Path:
    """Mask interior ocean/lagoon cells to NoData while keeping a 1-cell shoreline outlet ring."""
    if DEM_HYDRO_INPUT_UTM31N_PATH.exists() and not force:
        print(f"[cache] Using existing hydrology input DEM: {DEM_HYDRO_INPUT_UTM31N_PATH}")
        return DEM_HYDRO_INPUT_UTM31N_PATH

    with rasterio.open(DEM_UTM31N_PATH) as dem_src, rasterio.open(WBM_UTM31N_PATH) as wbm_src:
        dem = dem_src.read(1)
        wbm = wbm_src.read(1)
        profile = dem_src.profile.copy()

    # Ocean (1) and large lagoon/lake (2) water bodies in Copernicus WBM.
    # Reason: WhiteboxTools treats NoData cells as terminal boundary sinks during depression filling
    # and D8 routing. Eroding the water mask by 1 pixel leaves a 0.0m shoreline ring as the outlet
    # for coastal catchments while preventing synthetic parallel stream lines across open water.
    open_water = (wbm == 1) | (wbm == 2)
    interior_open_water = binary_erosion(open_water, iterations=1, border_value=1)

    hydro_dem = dem.copy()
    hydro_dem[interior_open_water] = DEM_NODATA

    with rasterio.open(DEM_HYDRO_INPUT_UTM31N_PATH, "w", **profile) as dst:
        dst.write(hydro_dem, 1)

    print(f"[process] Saved hydrologically masked DEM: {DEM_HYDRO_INPUT_UTM31N_PATH}")
    return DEM_HYDRO_INPUT_UTM31N_PATH


def compute_d8_hydrology(force: bool = False) -> tuple[Path, Path, Path, Path]:
    """Run depression filling, D8 flow pointer, D8 flow accumulation, and stream extraction."""
    wbt = get_wbt()

    if not DEM_FILLED_UTM31N_PATH.exists() or force:
        # Reason for fix_flats=True: Copernicus DEM over coastal floodplains contains flat zones
        # after pit filling; Garbrecht & Martz flat resolution enforces a tiny gradient toward outlets.
        ok = wbt.fill_depressions(
            dem=str(DEM_HYDRO_INPUT_UTM31N_PATH),
            output=str(DEM_FILLED_UTM31N_PATH),
            fix_flats=True,
        )
        if ok != 0 or not DEM_FILLED_UTM31N_PATH.exists():
            raise RuntimeError("WhiteboxTools fill_depressions failed.")
        print(f"[process] Saved depression-filled DEM: {DEM_FILLED_UTM31N_PATH}")
    else:
        print(f"[cache] Using existing depression-filled DEM: {DEM_FILLED_UTM31N_PATH}")

    if not D8_POINTER_UTM31N_PATH.exists() or force:
        ok = wbt.d8_pointer(
            dem=str(DEM_FILLED_UTM31N_PATH),
            output=str(D8_POINTER_UTM31N_PATH),
        )
        if ok != 0 or not D8_POINTER_UTM31N_PATH.exists():
            raise RuntimeError("WhiteboxTools d8_pointer failed.")
        print(f"[process] Saved D8 flow direction GeoTIFF: {D8_POINTER_UTM31N_PATH}")
    else:
        print(f"[cache] Using existing D8 flow direction GeoTIFF: {D8_POINTER_UTM31N_PATH}")

    if not D8_ACCUM_UTM31N_PATH.exists() or force:
        ok = wbt.d8_flow_accumulation(
            i=str(D8_POINTER_UTM31N_PATH),
            output=str(D8_ACCUM_UTM31N_PATH),
            out_type="cells",
            pntr=True,
        )
        if ok != 0 or not D8_ACCUM_UTM31N_PATH.exists():
            raise RuntimeError("WhiteboxTools d8_flow_accumulation failed.")
        print(f"[process] Saved D8 flow accumulation GeoTIFF: {D8_ACCUM_UTM31N_PATH}")
    else:
        print(f"[cache] Using existing D8 flow accumulation GeoTIFF: {D8_ACCUM_UTM31N_PATH}")

    if not STREAMS_RASTER_UTM31N_PATH.exists() or force:
        ok = wbt.extract_streams(
            flow_accum=str(D8_ACCUM_UTM31N_PATH),
            output=str(STREAMS_RASTER_UTM31N_PATH),
            threshold=float(FLOW_ACCUM_THRESHOLD_CELLS),
        )
        if ok != 0 or not STREAMS_RASTER_UTM31N_PATH.exists():
            raise RuntimeError("WhiteboxTools extract_streams failed.")
        print(
            f"[process] Saved extracted stream raster (threshold={FLOW_ACCUM_THRESHOLD_CELLS} cells): "
            f"{STREAMS_RASTER_UTM31N_PATH}"
        )
    else:
        print(f"[cache] Using existing stream raster: {STREAMS_RASTER_UTM31N_PATH}")

    return (
        DEM_FILLED_UTM31N_PATH,
        D8_POINTER_UTM31N_PATH,
        D8_ACCUM_UTM31N_PATH,
        STREAMS_RASTER_UTM31N_PATH,
    )


def vectorize_streams_to_geojson(force: bool = False) -> tuple[Path, Path]:
    """Trace D8 pointers along stream cells to create connected LineString GeoJSON files."""
    if STREAMS_GEOJSON_UTM31N_PATH.exists() and STREAMS_GEOJSON_WGS84_PATH.exists() and not force:
        print(f"[cache] Using existing stream GeoJSON files: {STREAMS_GEOJSON_UTM31N_PATH}")
        return STREAMS_GEOJSON_UTM31N_PATH, STREAMS_GEOJSON_WGS84_PATH

    wbt = get_wbt()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_shp = Path(tmpdir) / "streams.shp"
        ok = wbt.raster_streams_to_vector(
            streams=str(STREAMS_RASTER_UTM31N_PATH),
            d8_pntr=str(D8_POINTER_UTM31N_PATH),
            output=str(tmp_shp),
        )
        if ok != 0 or not tmp_shp.exists():
            raise RuntimeError("WhiteboxTools raster_streams_to_vector failed.")

        gdf_utm = gpd.read_file(tmp_shp)
        # Explicitly set CRS to EPSG:32631 (WhiteboxTools vector output omits .prj when tracing pointers).
        gdf_utm = gdf_utm.set_crs(UTM_CRS, allow_override=True)
        # Compute channel segment length strictly in projected metric CRS (EPSG:32631).
        gdf_utm["length_m"] = gdf_utm.geometry.length.round(1)
        gdf_utm = gdf_utm[gdf_utm["length_m"] > 0].copy()

        gdf_utm.to_file(STREAMS_GEOJSON_UTM31N_PATH, driver="GeoJSON")
        gdf_wgs84 = gdf_utm.to_crs(WGS84_CRS)
        gdf_wgs84.to_file(STREAMS_GEOJSON_WGS84_PATH, driver="GeoJSON")

    print(f"[process] Saved EPSG:32631 stream network GeoJSON: {STREAMS_GEOJSON_UTM31N_PATH}")
    print(f"[process] Saved EPSG:4326 stream network GeoJSON: {STREAMS_GEOJSON_WGS84_PATH}")
    return STREAMS_GEOJSON_UTM31N_PATH, STREAMS_GEOJSON_WGS84_PATH


def render_hydrology_preview(force: bool = False) -> Path:
    """Render a diagnostic PNG showing D8 flow accumulation and the extracted drainage network."""
    if HYDROLOGY_PNG_PATH.exists() and not force:
        print(f"[cache] Using existing hydrology preview PNG: {HYDROLOGY_PNG_PATH}")
        return HYDROLOGY_PNG_PATH

    with rasterio.open(DEM_UTM31N_PATH) as dem_src, \
         rasterio.open(HILLSHADE_UTM31N_PATH) as hs_src, \
         rasterio.open(D8_ACCUM_UTM31N_PATH) as acc_src, \
         rasterio.open(WBM_UTM31N_PATH) as wbm_src:
        dem = dem_src.read(1)
        hs = hs_src.read(1)
        acc = acc_src.read(1)
        acc_nodata = acc_src.nodata
        wbm = wbm_src.read(1)
        bounds = dem_src.bounds

    gdf_streams = gpd.read_file(STREAMS_GEOJSON_UTM31N_PATH)

    extent_km = [
        bounds.left / 1000.0,
        bounds.right / 1000.0,
        bounds.bottom / 1000.0,
        bounds.top / 1000.0,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), dpi=150, constrained_layout=True)

    # Left panel: D8 Flow Accumulation (log scale)
    ax0 = axes[0]
    ax0.imshow(hs, cmap="gray", extent=extent_km, origin="upper", vmin=40, vmax=245)
    valid_acc = (acc != acc_nodata) & np.isfinite(acc) & (acc >= 1.0)
    acc_plot = np.ma.masked_where(~valid_acc, acc)
    im0 = ax0.imshow(
        acc_plot,
        cmap="YlGnBu",
        norm=LogNorm(vmin=1.0, vmax=float(np.nanmax(acc[valid_acc]))),
        extent=extent_km,
        origin="upper",
        alpha=0.80,
    )
    water_mask = np.ma.masked_where(~((wbm == 1) | (wbm == 2)), np.ones_like(wbm))
    ax0.imshow(water_mask, cmap="Blues", extent=extent_km, origin="upper", vmin=0, vmax=1.4, alpha=0.55)
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.03, pad=0.02)
    cbar0.set_label("D8 Flow Accumulation (upstream cells, log scale)")
    ax0.set_title("D8 Flow Accumulation (EPSG:32631)", fontsize=12, fontweight="bold")
    ax0.set_xlabel("UTM Zone 31N Easting (km)")
    ax0.set_ylabel("UTM Zone 31N Northing (km)")

    # Right panel: Extracted Drainage Network over terrain
    ax1 = axes[1]
    ax1.imshow(hs, cmap="gray", extent=extent_km, origin="upper", vmin=40, vmax=245)
    ax1.imshow(
        dem,
        cmap="terrain",
        extent=extent_km,
        origin="upper",
        vmin=-2.0,
        vmax=90.0,
        alpha=0.42,
    )
    ax1.imshow(water_mask, cmap="Blues", extent=extent_km, origin="upper", vmin=0, vmax=1.4, alpha=0.65)

    # Scale stream coordinates to km to match axis extent_km
    gdf_km = gdf_streams.copy()
    gdf_km["geometry"] = gdf_km.geometry.scale(xfact=0.001, yfact=0.001, origin=(0, 0))
    gdf_km.plot(ax=ax1, color="#003399", linewidth=0.75, alpha=0.9)

    ax1.set_xlim(extent_km[0], extent_km[1])
    ax1.set_ylim(extent_km[2], extent_km[3])
    ax1.set_title(
        f"Extracted D8 Drainage Network (Threshold = {FLOW_ACCUM_THRESHOLD_CELLS:,} cells / 0.90 km²)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("UTM Zone 31N Easting (km)")
    ax1.set_ylabel("UTM Zone 31N Northing (km)")

    fig.savefig(HYDROLOGY_PNG_PATH)
    plt.close(fig)

    print(f"[process] Saved hydrology diagnostic PNG: {HYDROLOGY_PNG_PATH}")
    return HYDROLOGY_PNG_PATH


def print_summary() -> None:
    """Print summary statistics for depression filling, D8 accumulation, and extracted streams."""
    with rasterio.open(DEM_HYDRO_INPUT_UTM31N_PATH) as raw_src, \
         rasterio.open(DEM_FILLED_UTM31N_PATH) as fill_src, \
         rasterio.open(D8_ACCUM_UTM31N_PATH) as acc_src, \
         rasterio.open(STREAMS_RASTER_UTM31N_PATH) as str_src:
        raw_dem = raw_src.read(1)
        filled_dem = fill_src.read(1)
        acc = acc_src.read(1)
        streams = str_src.read(1)

        valid_land = (raw_dem != raw_src.nodata) & (filled_dem != fill_src.nodata)
        diff = filled_dem[valid_land] - raw_dem[valid_land]
        filled_cells = int(np.sum(diff > 1e-4))

        valid_acc = acc[(acc != acc_src.nodata) & np.isfinite(acc)]
        stream_cells = int(np.sum((streams != str_src.nodata) & (streams > 0)))

    gdf_streams = gpd.read_file(STREAMS_GEOJSON_UTM31N_PATH)
    total_km = gdf_streams["length_m"].sum() / 1000.0

    print("\n--- Milestone 2 Hydrology Summary ---")
    print(f"Conditioned land cells      : { int(np.sum(valid_land)):,}")
    print(
        f"Depression-filled cells     : {filled_cells:,} "
        f"({100.0 * filled_cells / np.sum(valid_land):.2f}% of land, max fill = {diff.max():.2f} m)"
    )
    print(f"Max D8 flow accumulation    : {int(valid_acc.max()):,} cells ({valid_acc.max() * 900 / 1e6:.1f} km²)")
    print(f"Stream threshold constant   : FLOW_ACCUM_THRESHOLD_CELLS = {FLOW_ACCUM_THRESHOLD_CELLS:,} cells (0.90 km²)")
    print(f"Extracted stream grid cells : {stream_cells:,}")
    print(f"Vector stream segments      : {len(gdf_streams):,} features ({total_km:.1f} km total channel length)")
    print(f"Stream raster (GeoTIFF)     : {STREAMS_RASTER_UTM31N_PATH}")
    print(f"Stream vector (UTM GeoJSON) : {STREAMS_GEOJSON_UTM31N_PATH}")
    print(f"Stream vector (WGS84)       : {STREAMS_GEOJSON_WGS84_PATH}")


def run_step2(force: bool = False) -> None:
    download_and_reproject_wbm(force=force)
    prepare_hydro_dem(force=force)
    compute_d8_hydrology(force=force)
    vectorize_streams_to_geojson(force=force)
    render_hydrology_preview(force=force)
    print_summary()


if __name__ == "__main__":
    run_step2()
