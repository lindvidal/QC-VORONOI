from pathlib import Path

# Paths — resolve root relative to this file's location (src/config.py → project root)
ROOT = Path(__file__).resolve().parent.parent
HDF5_PATH = ROOT / "2 - Organized data gauge" / "UNIPLU_DAILY_2014_2025.h5"
UNIPLU_DIR = ROOT / "src" / "data" / "UNIPLU"
RESULTS_DIR = ROOT / "src" / "results"
FIGURES_DIR = ROOT / "src" / "figures"
PARQUET_DIR = ROOT / "src" / "data" / "parquet"

# Rainfall event detection
MITS_MINUTES = [30, 360, 1439]   # Minimum Intra-event Time (30 min, 6h, 24h)
MINIMUM_DEPTH_MM = 1.0           # Minimum rainfall depth to consider an event
MIN_VALID_DAYS = 305             # Minimum days with data to process a station

# QC parameters (Delaunay)
P1_VALUES = [round(1.50 + 0.05 * i, 2) for i in range(7)]   # 1.50 to 1.80
P2_VALUES = [round(0.60 + 0.05 * i, 2) for i in range(7)]   # 0.60 to 0.90

# QC filters
ANNUAL_RAIN_MIN = 300
ANNUAL_RAIN_MAX = 3000
MAX_RAIN_EVENT_MM = 40
MIN_DAY_COUNT = 300

# Quality threshold: number of HQ factors needed for overall HQ classification
QUALITY_THRESHOLD = 3

# Quality factors used in hybrid QC
QUALITY_FACTORS = [
    "yearly_rainfall_quality",
    "rainfall_event_quality",
    "rainfall_intensity_quality",
    "rainfall_duration_quality",
]

# Visualization
FIGURE_DPI = 150
FIGURE_FORMAT = "png"
