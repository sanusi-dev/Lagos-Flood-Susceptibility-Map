"""Central configuration for CRS, bounding boxes, file paths, and thresholds."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Explicit Coordinate Reference Systems.
WGS84_CRS = "EPSG:4326"
UTM_CRS = "EPSG:32631"  # WGS 84 / UTM zone 31N (meters)

# Study area bounding box in EPSG:4326: (min_lon, min_lat, max_lon, max_lat)
LAGOS_BBOX_WGS84 = (3.0, 6.3, 4.0, 6.8)

# Target grid resolution in EPSG:32631 (meters) matching Copernicus GLO-30.
TARGET_RESOLUTION_M = 30.0
DEM_NODATA = -9999.0

# Public Copernicus GLO-30 AWS S3 bucket (no authentication required).
COPERNICUS_S3_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

# 1x1 degree tiles covering lon 3.0-4.0, lat 6.3-6.8 (including E004 for eastern boundary overlap).
COPERNICUS_TILES = [
    "Copernicus_DSM_COG_10_N06_00_E003_00_DEM",
    "Copernicus_DSM_COG_10_N06_00_E004_00_DEM",
]

# Milestone 1 output paths.
DEM_CLIPPED_WGS84_PATH = PROCESSED_DIR / "dem_clipped_wgs84.tif"
DEM_UTM31N_PATH = PROCESSED_DIR / "dem_utm31n.tif"
HILLSHADE_UTM31N_PATH = PROCESSED_DIR / "hillshade_utm31n.tif"
HILLSHADE_PNG_PATH = OUTPUTS_DIR / "lagos_hillshade.png"

# Milestone 2 hydrology constants and output paths.
# 1,000 cells * (30m * 30m) = 0.90 km^2 upstream contributing area threshold for stream initiation.
FLOW_ACCUM_THRESHOLD_CELLS = 1000

WBM_UTM31N_PATH = PROCESSED_DIR / "wbm_utm31n.tif"
DEM_HYDRO_INPUT_UTM31N_PATH = PROCESSED_DIR / "dem_hydro_input_utm31n.tif"
DEM_FILLED_UTM31N_PATH = PROCESSED_DIR / "dem_filled_utm31n.tif"
D8_POINTER_UTM31N_PATH = PROCESSED_DIR / "d8_pointer_utm31n.tif"
D8_ACCUM_UTM31N_PATH = PROCESSED_DIR / "d8_accum_utm31n.tif"
STREAMS_RASTER_UTM31N_PATH = PROCESSED_DIR / "streams_utm31n.tif"
STREAMS_GEOJSON_UTM31N_PATH = PROCESSED_DIR / "streams_utm31n.geojson"
STREAMS_GEOJSON_WGS84_PATH = OUTPUTS_DIR / "streams_wgs84.geojson"
HYDROLOGY_PNG_PATH = OUTPUTS_DIR / "lagos_hydrology.png"

# Milestone 3 HAND, OSM waterways, distance to drainage, and slope paths.
OSM_WATERWAYS_TAGS = {"waterway": ["river", "canal", "stream", "drain", "ditch"]}
OSM_WATERWAYS_WGS84_PATH = RAW_DIR / "osm_waterways_wgs84.geojson"
OSM_WATERWAYS_UTM31N_PATH = PROCESSED_DIR / "osm_waterways_utm31n.geojson"
DRAINAGE_TARGET_UTM31N_PATH = PROCESSED_DIR / "drainage_target_utm31n.tif"
HAND_UTM31N_PATH = PROCESSED_DIR / "hand_utm31n.tif"
DIST_DRAINAGE_UTM31N_PATH = PROCESSED_DIR / "dist_to_drainage_utm31n.tif"
SLOPE_UTM31N_PATH = PROCESSED_DIR / "slope_deg_utm31n.tif"
HAND_PNG_PATH = OUTPUTS_DIR / "lagos_hand_distance_slope.png"

# Milestone 4 Risk Classification Configuration (all thresholds in metric / degree units).
# Primary factor: Height Above Nearest Drainage (HAND) in meters.
HAND_HIGH_MAX_M = 2.0      # <= 2.0 m above receiving channel: high overbank/runoff susceptibility
HAND_MEDIUM_MAX_M = 5.0    # 2.0 - 5.0 m above receiving channel: moderate susceptibility

# Secondary factor 1: Euclidean distance to nearest drainage channel or OSM waterway (meters, EPSG:32631).
DIST_HIGH_MAX_M = 300.0    # <= 300 m from channel/canal: high proximity exposure
DIST_MEDIUM_MAX_M = 600.0  # 300 - 600 m from channel/canal: moderate proximity exposure

# Secondary factor 2: Terrain slope (degrees, EPSG:32631).
SLOPE_FLAT_MAX_DEG = 1.5      # <= 1.5 deg: flat terrain prone to surface water ponding
SLOPE_MODERATE_MAX_DEG = 3.0  # 1.5 - 3.0 deg: gentle slope with slow overland drainage

# Factor weights for composite susceptibility score (sum = 1.00, HAND is primary).
WEIGHT_HAND = 0.60
WEIGHT_DIST = 0.25
WEIGHT_SLOPE = 0.15

# Composite score cutoffs on the [1.0, 3.0] scale.
RISK_SCORE_HIGH_MIN = 2.40
RISK_SCORE_MEDIUM_MIN = 1.75

# Discrete raster class codes and display metadata.
RISK_CLASS_NODATA = 0
RISK_CLASS_LOW = 1
RISK_CLASS_MEDIUM = 2
RISK_CLASS_HIGH = 3

RISK_CLASS_LABELS = {
    RISK_CLASS_NODATA: "Water / NoData",
    RISK_CLASS_LOW: "Low",
    RISK_CLASS_MEDIUM: "Medium",
    RISK_CLASS_HIGH: "High",
}

# RGBA colors (0-255) for the web map transparent PNG overlay.
RISK_CLASS_COLORS_RGBA = {
    RISK_CLASS_NODATA: (0, 0, 0, 0),         # Fully transparent over ocean/lagoon
    RISK_CLASS_LOW: (44, 162, 95, 175),      # Green (#2ca25f)
    RISK_CLASS_MEDIUM: (254, 178, 76, 195),  # Amber/Orange (#feb24c)
    RISK_CLASS_HIGH: (227, 26, 28, 215),     # Red (#e31a1c)
}

RISK_UTM31N_PATH = PROCESSED_DIR / "risk_class_utm31n.tif"
RISK_WGS84_PATH = PROCESSED_DIR / "risk_class_wgs84.tif"
RISK_OVERLAY_PNG_PATH = OUTPUTS_DIR / "risk_overlay_wgs84.png"
RISK_ZONES_GEOJSON_PATH = OUTPUTS_DIR / "risk_zones_wgs84.geojson"
RISK_METADATA_JSON_PATH = OUTPUTS_DIR / "overlay_metadata.json"
RISK_MAP_PNG_PATH = OUTPUTS_DIR / "lagos_risk_map.png"

# Milestone 6 Validation paths.
VALIDATION_INPUT_CSV_PATH = DATA_DIR / "validation_points.csv"
VALIDATION_OUTPUT_CSV_PATH = OUTPUTS_DIR / "validation_results.csv"
VALIDATION_SUMMARY_JSON_PATH = OUTPUTS_DIR / "validation_summary.json"
