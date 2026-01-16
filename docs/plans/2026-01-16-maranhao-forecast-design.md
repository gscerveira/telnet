# Maranhão Seasonal Precipitation Forecast Design

## Overview

Generate seasonal precipitation forecasts for the state of Maranhão, Brazil using the TelNet library, with verification against observed 2025 data.

## Requirements

- **Target region:** Maranhão state, Brazil
- **Forecast periods:** All four 2025 initializations (January, April, July, October)
- **Lead times:** 1-6 months ahead
- **Outputs:** Probabilistic maps, raw ensemble data (20 members), verification metrics
- **Compute:** Google Colab GPU via SSH tunnel
- **Data source:** ERA5 from AWS Open Data (replacing CDS)

## Geographic Configuration

**Bounding box:**
```
Latitude:  -10.25°S to -1.0°S
Longitude: -48.75°W to -41.5°W
```

**Shapefile:** IBGE state boundary for precise masking

## Data Sources

| Data | Source | Access Method |
|------|--------|---------------|
| ERA5 precipitation, winds, geopotential | AWS Open Data (`s3://era5-pds`) | `xarray` + `s3fs` |
| ERSSTv5 sea surface temperature | NOAA PSL | Direct HTTP download |
| Maranhão shapefile | IBGE | Direct download |
| Observed precipitation (verification) | ERA5 on AWS | Same as above |

## Environment Setup

### Colab SSH Tunnel

1. Run setup notebook that installs `cloudflared` and starts SSH server
2. Connect from local terminal via tunnel URL
3. Full bash shell access with GPU (T4/A100)

### Dependencies

```
# Additional packages beyond TelNet requirements
s3fs          # AWS S3 access
zarr          # Cloud-optimized data format
cartopy       # Geographic plotting
geopandas     # Shapefile handling
```

### Persistence Strategy

- Mount Google Drive for checkpoints and data
- Use git for code changes
- Scripts designed to be resumable (skip completed steps)

## Workflow

### Phase 1: One-time Setup

1. **Create Colab SSH notebook** - Installs cloudflared, starts SSH tunnel
2. **Connect via SSH** - Run notebook, copy tunnel URL, connect from terminal
3. **Clone repo and install dependencies** - git clone, pip install extras, mount Drive
4. **Create AWS data downloader** - New module replacing CDS with AWS ERA5 access

### Phase 2: Data Acquisition

1. **Download ERA5 from AWS** - Precipitation, winds, geopotential (1940-2025)
2. **Download ERSSTv5** - Sea surface temperature from NOAA
3. **Compute climate indices** - Run existing `compute_climate_indices.py`
4. **Download Maranhão shapefile** - From IBGE geodata service

### Phase 3: Model Training

1. **Feature pre-selection** - Run `feature_pre_selection.py` to identify best predictors
2. **Model selection** - Grid search over hyperparameters with GPU
3. **Model testing** - Train final model, evaluate on test set

### Phase 4: Generate 2025 Forecasts

1. **Run forecasts** for each initialization:
   - January 2025 (forecasts Feb-Jul 2025)
   - April 2025 (forecasts May-Oct 2025)
   - July 2025 (forecasts Aug-Jan 2026)
   - October 2025 (forecasts Nov-Apr 2026)
2. **Extract Maranhão region** - Clip to state boundaries
3. **Generate probability maps** - Tercile probabilities, variable importance

### Phase 5: Verification

1. **Download observed 2025 precipitation** - ERA5 actuals
2. **Compute skill metrics** - RPSS, reliability diagrams, rank histograms
3. **Export final results** - Sync to Drive / download locally

## Output Structure

```
results/
├── forecasts/
│   ├── 202501/                   # January 2025 initialization
│   │   ├── ensemble_raw.nc       # 20-member ensemble data
│   │   ├── tercile_probs.nc      # Below/Normal/Above probabilities
│   │   └── maps/                 # PNG visualizations
│   ├── 202504/                   # April 2025
│   ├── 202507/                   # July 2025
│   └── 202510/                   # October 2025
└── verification/
    ├── skill_scores.csv          # RPSS, RPS by lead time
    └── reliability_diagrams.png  # Calibration assessment
```

## Files to Create

| File | Purpose |
|------|---------|
| `colab_ssh_setup.ipynb` | Colab notebook for SSH tunnel setup |
| `download_era5_aws.py` | ERA5 downloader using AWS S3 |
| `extract_maranhao.py` | Clips outputs to Maranhão boundaries |
| `run_maranhao_forecast.py` | Orchestrates the full workflow |
| `verify_forecasts.py` | Compares forecasts against observations |
| `data/lat_lon_boundaries.txt` | Maranhão bounding box |
| `data/maranhao.geojson` | State boundary for masking |

## Existing Files (Unchanged)

- `compute_climate_indices.py`
- `feature_pre_selection.py`
- `model_selection.py`
- `model_testing.py`
- `generate_forecast.py`
- All model architecture code (`models/`, `modules/`)
