# Maranhão Seasonal Forecast Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate seasonal precipitation forecasts for Maranhão state (Brazil) for all 2025 initialization periods, with verification against observations.

**Architecture:** Replace CDS data source with AWS Open Data for ERA5, add Maranhão-specific extraction and masking utilities, create orchestration script for the full workflow, run on Google Colab via SSH tunnel.

**Tech Stack:** Python 3.10+, PyTorch, xarray, s3fs, zarr, cartopy, geopandas

---

## Task 1: Create Colab SSH Setup Notebook

**Files:**
- Create: `colab_ssh_setup.ipynb`

**Step 1: Create the notebook file**

Create a Jupyter notebook with the following cells:

```python
# Cell 1 - Install cloudflared
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb

# Cell 2 - Set root password and start SSH
import subprocess
import random
import string

password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
subprocess.run(['chpasswd'], input=f'root:{password}'.encode())
subprocess.run(['service', 'ssh', 'start'])
print(f"Root password: {password}")

# Cell 3 - Start cloudflared tunnel
!cloudflared tunnel --url ssh://localhost:22 2>&1 | grep -o 'https://[^.]*\.trycloudflare\.com'
```

**Step 2: Verify notebook structure**

Run: `python -c "import json; json.load(open('colab_ssh_setup.ipynb'))"`
Expected: No output (valid JSON)

**Step 3: Commit**

```bash
git add colab_ssh_setup.ipynb
git commit -m "feat: add Colab SSH tunnel setup notebook"
```

---

## Task 2: Create AWS ERA5 Data Downloader

**Files:**
- Create: `download_era5_aws.py`

**Step 1: Create the AWS downloader module**

```python
"""
Download ERA5 data from AWS Open Data Registry instead of CDS.
Provides the same output format as download_preprocess_data.py but uses
s3://era5-pds which requires no authentication.
"""

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import s3fs
from calendar import monthrange
from utils import exp_data_dir


def get_era5_aws_client():
    """Create S3 filesystem client for anonymous access."""
    return s3fs.S3FileSystem(anon=True)


def download_era5_precipitation_aws(init_date, final_date, lats, lons):
    """
    Download ERA5 monthly precipitation from AWS Open Data.

    AWS ERA5 stores hourly data - we aggregate to monthly totals.
    """
    root_datadir = os.getenv('TELNET_DATADIR')
    era5_dir = os.path.join(root_datadir, 'era5')
    os.makedirs(era5_dir, exist_ok=True)

    iyr = int(init_date[:4])
    fyr = int(final_date[:4])

    output_file = os.path.join(era5_dir, f'era5_pr_{iyr}-{fyr}_preprocessed.nc')
    if os.path.exists(output_file):
        print(f"ERA5 precipitation already exists: {output_file}")
        return

    fs = get_era5_aws_client()
    monthly_data = []

    for year in range(iyr, fyr + 1):
        for month in range(1, 13):
            if year == iyr and month < int(init_date[5:7]):
                continue
            if year == fyr and month > int(final_date[5:7]):
                continue

            zarr_path = f's3://era5-pds/{year}/{month:02d}/data/precipitation_amount_1hour_Accumulation.zarr'

            try:
                print(f"Downloading precipitation for {year}-{month:02d}...")
                ds = xr.open_zarr(fs.get_mapper(zarr_path))

                # Subset to region (with buffer for interpolation)
                lat_buffer = 5
                lon_buffer = 5
                ds = ds.sel(
                    lat=slice(lats[1] + lat_buffer, lats[0] - lat_buffer),
                    lon=slice(lons[0] - lon_buffer + 360 if lons[0] < 0 else lons[0] - lon_buffer,
                              lons[1] + lon_buffer + 360 if lons[1] < 0 else lons[1] + lon_buffer)
                )

                # Sum to monthly total
                monthly_total = ds['precipitation_amount_1hour_Accumulation'].sum(dim='time0')
                monthly_total = monthly_total.expand_dims(
                    time=[np.datetime64(f'{year}-{month:02d}-01')]
                )
                monthly_data.append(monthly_total)

            except Exception as e:
                print(f"Warning: Could not load {year}-{month:02d}: {e}")
                continue

    if monthly_data:
        combined = xr.concat(monthly_data, dim='time')
        combined = combined.rename({'precipitation_amount_1hour_Accumulation': 'pr'})

        # Convert longitude from 0-360 to -180-180
        combined = combined.assign_coords(
            lon=(((combined.lon + 180) % 360) - 180)
        ).sortby('lon')

        # Final subset to exact region
        combined = combined.sel(
            lat=slice(lats[1], lats[0]),
            lon=slice(lons[0], lons[1])
        )

        combined.to_netcdf(output_file)
        print(f"Saved precipitation to {output_file}")


def download_era5_winds_aws(init_date, final_date):
    """Download ERA5 10m wind components from AWS."""
    root_datadir = os.getenv('TELNET_DATADIR')
    era5_dir = os.path.join(root_datadir, 'era5')
    os.makedirs(era5_dir, exist_ok=True)

    iyr = int(init_date[:4])
    fyr = int(final_date[:4])

    fs = get_era5_aws_client()

    for var_name, aws_var in [('u10', 'eastward_wind_at_10_metres'),
                               ('v10', 'northward_wind_at_10_metres')]:
        output_file = os.path.join(era5_dir, f'era5_{var_name}_{iyr}-{fyr}_preprocessed.nc')
        if os.path.exists(output_file):
            print(f"ERA5 {var_name} already exists: {output_file}")
            continue

        monthly_data = []

        for year in range(iyr, fyr + 1):
            for month in range(1, 13):
                if year == iyr and month < int(init_date[5:7]):
                    continue
                if year == fyr and month > int(final_date[5:7]):
                    continue

                zarr_path = f's3://era5-pds/{year}/{month:02d}/data/{aws_var}.zarr'

                try:
                    print(f"Downloading {var_name} for {year}-{month:02d}...")
                    ds = xr.open_zarr(fs.get_mapper(zarr_path))

                    # Monthly mean
                    monthly_mean = ds[aws_var].mean(dim='time0')
                    monthly_mean = monthly_mean.expand_dims(
                        time=[np.datetime64(f'{year}-{month:02d}-01')]
                    )
                    monthly_data.append(monthly_mean)

                except Exception as e:
                    print(f"Warning: Could not load {year}-{month:02d}: {e}")
                    continue

        if monthly_data:
            combined = xr.concat(monthly_data, dim='time')
            combined = combined.rename({aws_var: var_name})

            # Interpolate to 2-degree grid
            lon2interp = np.arange(0., 360., 2.0)
            lat2interp = np.arange(-88., 90., 2.0)[::-1]
            combined = combined.interp(lat=lat2interp, lon=lon2interp, method='linear')

            combined.to_netcdf(output_file)
            print(f"Saved {var_name} to {output_file}")


def download_era5_geopotential_aws(init_date, final_date):
    """Download ERA5 geopotential heights from AWS."""
    root_datadir = os.getenv('TELNET_DATADIR')
    era5_dir = os.path.join(root_datadir, 'era5')
    os.makedirs(era5_dir, exist_ok=True)

    iyr = int(init_date[:4])
    fyr = int(final_date[:4])

    output_file = os.path.join(era5_dir, f'era5_hgt_{iyr}-{fyr}_preprocessed.nc')
    if os.path.exists(output_file):
        print(f"ERA5 geopotential already exists: {output_file}")
        return

    fs = get_era5_aws_client()
    levels = [500, 700, 1000]
    level_data = {level: [] for level in levels}

    for year in range(iyr, fyr + 1):
        for month in range(1, 13):
            if year == iyr and month < int(init_date[5:7]):
                continue
            if year == fyr and month > int(final_date[5:7]):
                continue

            for level in levels:
                zarr_path = f's3://era5-pds/{year}/{month:02d}/data/geopotential_at_{level}hPa.zarr'

                try:
                    print(f"Downloading geopotential {level}hPa for {year}-{month:02d}...")
                    ds = xr.open_zarr(fs.get_mapper(zarr_path))

                    var_name = f'geopotential_at_{level}hPa'
                    monthly_mean = ds[var_name].mean(dim='time0')
                    # Convert geopotential to geopotential height
                    monthly_mean = monthly_mean / 9.80665
                    monthly_mean = monthly_mean.expand_dims(
                        time=[np.datetime64(f'{year}-{month:02d}-01')]
                    )
                    level_data[level].append(monthly_mean)

                except Exception as e:
                    print(f"Warning: Could not load {level}hPa {year}-{month:02d}: {e}")
                    continue

    # Combine all levels
    combined_levels = []
    for level in levels:
        if level_data[level]:
            combined = xr.concat(level_data[level], dim='time')
            combined = combined.expand_dims(level=[level])
            combined_levels.append(combined)

    if combined_levels:
        final = xr.concat(combined_levels, dim='level')
        final = final.rename({f'geopotential_at_{levels[0]}hPa': 'height'})

        # Interpolate to 2-degree grid
        lon2interp = np.arange(0., 360., 2.0)
        lat2interp = np.arange(-88., 90., 2.0)[::-1]
        final = final.interp(lat=lat2interp, lon=lon2interp, method='linear')

        final.to_netcdf(output_file)
        print(f"Saved geopotential to {output_file}")


def download_era5_land_sea_mask_aws(lats, lons):
    """Download ERA5 land-sea mask from AWS."""
    root_datadir = os.getenv('TELNET_DATADIR')
    era5_dir = os.path.join(root_datadir, 'era5')
    os.makedirs(era5_dir, exist_ok=True)

    output_file = os.path.join(era5_dir, 'era5_land_sea_mask_preprocessed.nc')
    if os.path.exists(output_file):
        print(f"Land-sea mask already exists: {output_file}")
        return

    fs = get_era5_aws_client()

    # Land-sea mask is static, grab from any month
    zarr_path = 's3://era5-pds/2020/01/data/land_sea_mask.zarr'

    try:
        print("Downloading land-sea mask...")
        ds = xr.open_zarr(fs.get_mapper(zarr_path))

        # Get first time step (mask is static)
        mask = ds['land_sea_mask'].isel(time0=0)

        # Convert longitude from 0-360 to -180-180
        mask = mask.assign_coords(
            lon=(((mask.lon + 180) % 360) - 180)
        ).sortby('lon')

        # Subset to region
        mask = mask.sel(
            lat=slice(lats[1], lats[0]),
            lon=slice(lons[0], lons[1])
        )

        mask = mask.expand_dims(time=[np.datetime64('2020-01-01')])
        mask.to_netcdf(output_file)
        print(f"Saved land-sea mask to {output_file}")

    except Exception as e:
        print(f"Error downloading land-sea mask: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Download ERA5 data from AWS Open Data for TelNet'
    )
    parser.add_argument(
        '-idate', '--initdate',
        help='Initial date YYYYMM (e.g., 194001)',
        required=True
    )
    parser.add_argument(
        '-fdate', '--finaldate',
        help='Final date YYYYMM (e.g., 202512)',
        required=True
    )

    args = parser.parse_args()

    init_date = f'{args.initdate[:4]}-{args.initdate[4:6]}-01'
    final_date = f'{args.finaldate[:4]}-{args.finaldate[4:6]}-01'

    # Load or create region bounds
    bounds_file = os.path.join(exp_data_dir, 'lat_lon_boundaries.txt')
    if os.path.exists(bounds_file):
        with open(bounds_file, 'r') as f:
            lines = f.readlines()
            lats = [float(lines[0].strip()), float(lines[1].strip())]
            lons = [float(lines[2].strip()), float(lines[3].strip())]
    else:
        # Default to Maranhão region with buffer
        lats = [-10.25, -1.0]
        lons = [-48.75, -41.5]
        os.makedirs(exp_data_dir, exist_ok=True)
        with open(bounds_file, 'w') as f:
            f.write(f"{lats[0]}\n{lats[1]}\n{lons[0]}\n{lons[1]}\n")

    print(f"Downloading ERA5 data for region: lat={lats}, lon={lons}")
    print(f"Period: {init_date} to {final_date}")

    # Download all variables
    download_era5_precipitation_aws(init_date, final_date, lats, lons)
    download_era5_winds_aws(init_date, final_date)
    download_era5_geopotential_aws(init_date, final_date)
    download_era5_land_sea_mask_aws(lats, lons)

    print("\nERA5 download complete!")


if __name__ == "__main__":
    main()
```

**Step 2: Verify Python syntax**

Run: `python -m py_compile download_era5_aws.py`
Expected: No output (valid syntax)

**Step 3: Commit**

```bash
git add download_era5_aws.py
git commit -m "feat: add AWS-based ERA5 downloader replacing CDS"
```

---

## Task 3: Create Maranhão Bounds Configuration

**Files:**
- Create: `configs/maranhao_bounds.txt`

**Step 1: Create config directory and bounds file**

```bash
mkdir -p configs
```

Create `configs/maranhao_bounds.txt`:
```
-10.25
-1.0
-48.75
-41.5
```

**Step 2: Commit**

```bash
git add configs/maranhao_bounds.txt
git commit -m "feat: add Maranhão geographic bounds configuration"
```

---

## Task 4: Create Maranhão Shapefile Downloader

**Files:**
- Create: `download_maranhao_shapefile.py`

**Step 1: Create the shapefile downloader**

```python
"""
Download Maranhão state boundary shapefile from IBGE.
"""

import os
import zipfile
import urllib.request
import geopandas as gpd


def download_maranhao_shapefile(output_dir):
    """
    Download and extract Maranhão state boundary from IBGE.
    Returns path to the shapefile.
    """
    os.makedirs(output_dir, exist_ok=True)

    shapefile_path = os.path.join(output_dir, 'maranhao.shp')
    if os.path.exists(shapefile_path):
        print(f"Shapefile already exists: {shapefile_path}")
        return shapefile_path

    # IBGE provides state boundaries
    # Using simplified boundaries for efficiency
    url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/MA/MA_UF_2022.zip"

    zip_path = os.path.join(output_dir, 'ma_uf.zip')

    print("Downloading Maranhão shapefile from IBGE...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_dir)

    # Find the extracted shapefile
    for f in os.listdir(output_dir):
        if f.endswith('.shp') and 'UF' in f:
            extracted_shp = os.path.join(output_dir, f)
            # Rename to standard name
            gdf = gpd.read_file(extracted_shp)
            gdf.to_file(shapefile_path)
            print(f"Saved to {shapefile_path}")
            break

    # Cleanup
    os.remove(zip_path)

    return shapefile_path


def create_maranhao_mask(shapefile_path, lats, lons, resolution=0.25):
    """
    Create a boolean mask array for Maranhão from the shapefile.
    """
    import numpy as np
    from shapely.geometry import Point

    gdf = gpd.read_file(shapefile_path)
    geometry = gdf.unary_union

    lat_grid = np.arange(lats[0], lats[1], resolution)
    lon_grid = np.arange(lons[0], lons[1], resolution)

    mask = np.zeros((len(lat_grid), len(lon_grid)), dtype=bool)

    for i, lat in enumerate(lat_grid):
        for j, lon in enumerate(lon_grid):
            point = Point(lon, lat)
            if geometry.contains(point):
                mask[i, j] = True

    return mask, lat_grid, lon_grid


if __name__ == "__main__":
    output_dir = os.path.join(os.getenv('TELNET_DATADIR', 'data'), 'shapefiles')
    download_maranhao_shapefile(output_dir)
```

**Step 2: Verify Python syntax**

Run: `python -m py_compile download_maranhao_shapefile.py`
Expected: No output (valid syntax)

**Step 3: Commit**

```bash
git add download_maranhao_shapefile.py
git commit -m "feat: add Maranhão shapefile downloader from IBGE"
```

---

## Task 5: Create Maranhão Extraction Module

**Files:**
- Create: `extract_maranhao.py`

**Step 1: Create the extraction module**

```python
"""
Extract and mask TelNet outputs to Maranhão state boundaries.
"""

import os
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import mapping
import rioxarray  # Enables rio accessor for xarray


def load_maranhao_geometry(shapefile_dir):
    """Load Maranhão boundary geometry."""
    shapefile_path = os.path.join(shapefile_dir, 'maranhao.shp')
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(
            f"Maranhão shapefile not found at {shapefile_path}. "
            "Run download_maranhao_shapefile.py first."
        )
    gdf = gpd.read_file(shapefile_path)
    return gdf


def extract_maranhao_region(ds, shapefile_dir, var_name=None):
    """
    Clip xarray Dataset/DataArray to Maranhão boundaries.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input data with lat/lon coordinates
    shapefile_dir : str
        Directory containing maranhao.shp
    var_name : str, optional
        Variable name if ds is Dataset

    Returns
    -------
    xr.Dataset or xr.DataArray
        Data clipped to Maranhão boundaries
    """
    gdf = load_maranhao_geometry(shapefile_dir)

    # Ensure CRS is set
    if ds.rio.crs is None:
        ds = ds.rio.write_crs("EPSG:4326")

    # Set spatial dimensions
    if 'lat' in ds.dims and 'lon' in ds.dims:
        ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')

    # Clip to geometry
    clipped = ds.rio.clip(gdf.geometry, gdf.crs, drop=True)

    return clipped


def mask_outside_maranhao(ds, shapefile_dir):
    """
    Mask values outside Maranhão with NaN (keeps rectangular grid).

    Useful when you need the full grid but want to hide non-Maranhão areas.
    """
    gdf = load_maranhao_geometry(shapefile_dir)
    geometry = gdf.unary_union

    # Create mask
    lats = ds.lat.values
    lons = ds.lon.values

    from shapely.geometry import Point
    mask = np.zeros((len(lats), len(lons)), dtype=bool)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if geometry.contains(Point(lon, lat)):
                mask[i, j] = True

    # Apply mask
    mask_da = xr.DataArray(mask, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})

    if isinstance(ds, xr.Dataset):
        for var in ds.data_vars:
            ds[var] = ds[var].where(mask_da)
    else:
        ds = ds.where(mask_da)

    return ds


def extract_forecast_for_maranhao(forecast_path, output_path, shapefile_dir):
    """
    Extract forecast data for Maranhão and save to new file.

    Parameters
    ----------
    forecast_path : str
        Path to forecast NetCDF file
    output_path : str
        Path to save extracted data
    shapefile_dir : str
        Directory containing maranhao.shp
    """
    print(f"Loading forecast from {forecast_path}...")
    ds = xr.open_dataset(forecast_path)

    print("Extracting Maranhão region...")
    ds_ma = mask_outside_maranhao(ds, shapefile_dir)

    print(f"Saving to {output_path}...")
    ds_ma.to_netcdf(output_path)

    print("Done!")
    return ds_ma


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract Maranhão region from forecast')
    parser.add_argument('input', help='Input forecast NetCDF file')
    parser.add_argument('output', help='Output NetCDF file')
    parser.add_argument('--shapefile-dir', default=None,
                        help='Directory containing maranhao.shp')

    args = parser.parse_args()

    shapefile_dir = args.shapefile_dir or os.path.join(
        os.getenv('TELNET_DATADIR', 'data'), 'shapefiles'
    )

    extract_forecast_for_maranhao(args.input, args.output, shapefile_dir)
```

**Step 2: Verify Python syntax**

Run: `python -m py_compile extract_maranhao.py`
Expected: No output (valid syntax)

**Step 3: Commit**

```bash
git add extract_maranhao.py
git commit -m "feat: add Maranhão extraction and masking utilities"
```

---

## Task 6: Create Forecast Verification Module

**Files:**
- Create: `verify_forecasts.py`

**Step 1: Create the verification module**

```python
"""
Verify TelNet forecasts against observed ERA5 precipitation.
Computes skill scores and generates reliability diagrams.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_observations(obs_path):
    """Load observed precipitation data."""
    return xr.open_dataset(obs_path)


def load_forecast_ensemble(forecast_path):
    """Load forecast ensemble data."""
    return xr.open_dataset(forecast_path)


def compute_tercile_categories(data, climatology):
    """
    Compute tercile categories (below/normal/above) based on climatology.

    Returns integer array: 0=below, 1=normal, 2=above
    """
    lower_tercile = climatology.quantile(1/3, dim='time')
    upper_tercile = climatology.quantile(2/3, dim='time')

    categories = xr.where(data < lower_tercile, 0,
                 xr.where(data > upper_tercile, 2, 1))
    return categories


def compute_tercile_probabilities(ensemble):
    """
    Compute tercile probabilities from ensemble members.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble forecast with 'member' dimension

    Returns
    -------
    xr.Dataset
        Dataset with prob_below, prob_normal, prob_above variables
    """
    climatology = ensemble.mean(dim='member')
    lower_tercile = climatology.quantile(1/3)
    upper_tercile = climatology.quantile(2/3)

    prob_below = (ensemble < lower_tercile).mean(dim='member')
    prob_above = (ensemble > upper_tercile).mean(dim='member')
    prob_normal = 1 - prob_below - prob_above

    return xr.Dataset({
        'prob_below': prob_below,
        'prob_normal': prob_normal,
        'prob_above': prob_above
    })


def compute_rps(forecast_probs, observed_category):
    """
    Compute Ranked Probability Score.

    Parameters
    ----------
    forecast_probs : xr.Dataset
        Dataset with prob_below, prob_normal, prob_above
    observed_category : xr.DataArray
        Observed category (0, 1, or 2)

    Returns
    -------
    xr.DataArray
        RPS values (lower is better, 0 is perfect)
    """
    # Cumulative probabilities
    cum_forecast = xr.concat([
        forecast_probs['prob_below'],
        forecast_probs['prob_below'] + forecast_probs['prob_normal'],
        xr.ones_like(forecast_probs['prob_below'])
    ], dim='category')

    # Cumulative observed (step function)
    cum_observed = xr.concat([
        (observed_category >= 1).astype(float),
        (observed_category >= 2).astype(float),
        xr.ones_like(observed_category, dtype=float)
    ], dim='category')

    rps = ((cum_forecast - cum_observed) ** 2).sum(dim='category') / 2
    return rps


def compute_rpss(rps, rps_climatology):
    """
    Compute Ranked Probability Skill Score.

    RPSS = 1 - RPS/RPS_clim

    RPSS > 0: better than climatology
    RPSS = 0: same as climatology
    RPSS < 0: worse than climatology
    """
    return 1 - rps / rps_climatology


def compute_climatology_rps():
    """RPS for climatological forecast (1/3, 1/3, 1/3)."""
    return 2/9  # Analytical result for tercile climatology


def reliability_diagram(forecast_probs, observed_binary, n_bins=10, ax=None):
    """
    Create reliability diagram for probabilistic forecasts.

    Parameters
    ----------
    forecast_probs : array-like
        Forecast probabilities [0, 1]
    observed_binary : array-like
        Binary observations (0 or 1)
    n_bins : int
        Number of probability bins
    ax : matplotlib axis, optional

    Returns
    -------
    dict
        Reliability statistics
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    forecast_probs = np.array(forecast_probs).flatten()
    observed_binary = np.array(observed_binary).flatten()

    # Remove NaN
    valid = ~(np.isnan(forecast_probs) | np.isnan(observed_binary))
    forecast_probs = forecast_probs[valid]
    observed_binary = observed_binary[valid]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    observed_freq = []
    forecast_freq = []
    counts = []

    for i in range(n_bins):
        mask = (forecast_probs >= bin_edges[i]) & (forecast_probs < bin_edges[i+1])
        if mask.sum() > 0:
            observed_freq.append(observed_binary[mask].mean())
            forecast_freq.append(forecast_probs[mask].mean())
            counts.append(mask.sum())
        else:
            observed_freq.append(np.nan)
            forecast_freq.append(bin_centers[i])
            counts.append(0)

    # Plot
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect reliability')
    ax.plot(forecast_freq, observed_freq, 'bo-', label='Forecast')
    ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.1, color='gray')

    ax.set_xlabel('Forecast probability')
    ax.set_ylabel('Observed frequency')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_aspect('equal')

    return {
        'bin_centers': bin_centers,
        'forecast_freq': forecast_freq,
        'observed_freq': observed_freq,
        'counts': counts
    }


def verify_forecast(forecast_path, obs_path, output_dir, shapefile_dir=None):
    """
    Run full verification suite for a forecast.

    Parameters
    ----------
    forecast_path : str
        Path to forecast NetCDF (with ensemble members)
    obs_path : str
        Path to observed precipitation NetCDF
    output_dir : str
        Directory to save verification outputs
    shapefile_dir : str, optional
        If provided, mask to Maranhão
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    forecast = xr.open_dataset(forecast_path)
    obs = xr.open_dataset(obs_path)

    # Optionally mask to Maranhão
    if shapefile_dir:
        from extract_maranhao import mask_outside_maranhao
        forecast = mask_outside_maranhao(forecast, shapefile_dir)
        obs = mask_outside_maranhao(obs, shapefile_dir)

    print("Computing tercile probabilities...")
    # Assuming forecast has 'pr' variable with 'member' dimension
    probs = compute_tercile_probabilities(forecast['pr'])

    print("Computing observed categories...")
    # Use forecast climatology for tercile boundaries
    obs_category = compute_tercile_categories(obs['pr'], forecast['pr'].mean(dim='member'))

    print("Computing RPS...")
    rps = compute_rps(probs, obs_category)
    rps_clim = compute_climatology_rps()
    rpss = compute_rpss(rps, rps_clim)

    # Spatial mean statistics
    mean_rps = float(rps.mean())
    mean_rpss = float(rpss.mean())

    print(f"\nVerification Results:")
    print(f"  Mean RPS:  {mean_rps:.4f}")
    print(f"  Mean RPSS: {mean_rpss:.4f}")

    # Save statistics
    stats_df = pd.DataFrame({
        'metric': ['RPS', 'RPSS'],
        'value': [mean_rps, mean_rpss]
    })
    stats_df.to_csv(os.path.join(output_dir, 'skill_scores.csv'), index=False)

    # Create reliability diagrams
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (cat, name) in enumerate([('below', 'Below Normal'),
                                      ('normal', 'Normal'),
                                      ('above', 'Above Normal')]):
        prob_key = f'prob_{cat}'
        obs_binary = (obs_category == i).values
        reliability_diagram(probs[prob_key].values, obs_binary, ax=axes[i])
        axes[i].set_title(f'{name} Reliability')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reliability_diagrams.png'), dpi=150)
    plt.close()

    print(f"\nOutputs saved to {output_dir}")

    return {
        'rps': mean_rps,
        'rpss': mean_rpss,
        'rps_spatial': rps,
        'rpss_spatial': rpss
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Verify forecasts against observations')
    parser.add_argument('forecast', help='Forecast NetCDF file')
    parser.add_argument('observations', help='Observations NetCDF file')
    parser.add_argument('--output-dir', '-o', default='verification',
                        help='Output directory')
    parser.add_argument('--shapefile-dir', default=None,
                        help='Shapefile directory for Maranhão masking')

    args = parser.parse_args()

    verify_forecast(args.forecast, args.observations, args.output_dir, args.shapefile_dir)
```

**Step 2: Verify Python syntax**

Run: `python -m py_compile verify_forecasts.py`
Expected: No output (valid syntax)

**Step 3: Commit**

```bash
git add verify_forecasts.py
git commit -m "feat: add forecast verification with skill scores and reliability diagrams"
```

---

## Task 7: Create Main Workflow Orchestrator

**Files:**
- Create: `run_maranhao_forecast.py`

**Step 1: Create the orchestrator script**

```python
"""
Main orchestrator for running TelNet seasonal forecasts for Maranhão.

This script coordinates the full workflow:
1. Download data (ERA5 from AWS, ERSSTv5 from NOAA)
2. Compute climate indices
3. Run feature pre-selection
4. Run model selection
5. Generate forecasts for specified initialization dates
6. Extract Maranhão region
7. Verify against observations

Usage:
    python run_maranhao_forecast.py --init-dates 202501 202504 202507 202510
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime


def run_command(cmd, description):
    """Run a shell command with logging."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"COMPLETED: {description}")
    return result


def setup_environment():
    """Verify environment variables are set."""
    required_vars = ['TELNET_DATADIR']

    for var in required_vars:
        if not os.getenv(var):
            print(f"ERROR: Environment variable {var} not set")
            print(f"Please set it: export {var}=/path/to/data")
            sys.exit(1)

    # Create data directories
    datadir = os.getenv('TELNET_DATADIR')
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(os.path.join(datadir, 'era5'), exist_ok=True)
    os.makedirs(os.path.join(datadir, 'shapefiles'), exist_ok=True)
    os.makedirs(os.path.join(datadir, 'models'), exist_ok=True)

    print(f"Data directory: {datadir}")


def download_data(init_date='194001', final_date='202512'):
    """Download all required data."""

    # Download ERA5 from AWS
    run_command(
        f"python download_era5_aws.py -idate {init_date} -fdate {final_date}",
        "Download ERA5 from AWS"
    )

    # Download ERSSTv5 (uses existing script logic)
    run_command(
        f"python -c \"from download_preprocess_data import download_ersstv5; download_ersstv5('{init_date[:4]}-{init_date[4:]}-01', '{final_date[:4]}-{final_date[4:]}-01')\"",
        "Download ERSSTv5 from NOAA"
    )

    # Download Maranhão shapefile
    run_command(
        "python download_maranhao_shapefile.py",
        "Download Maranhão shapefile"
    )


def compute_indices(final_date='202512'):
    """Compute climate indices."""
    run_command(
        f"python compute_climate_indices.py -fdate {final_date}",
        "Compute climate indices"
    )


def run_feature_selection(n_samples=100):
    """Run feature pre-selection."""
    run_command(
        f"python feature_pre_selection.py -n {n_samples}",
        "Feature pre-selection"
    )


def run_model_selection(n_samples=100, n_gpus=1):
    """Run model selection grid search."""
    run_command(
        f"./model_selection.sh {n_samples} {n_gpus}",
        "Model selection"
    )


def run_model_testing(n_samples=100, config=1):
    """Run model testing with selected configuration."""
    run_command(
        f"python model_testing.py -n {n_samples} -c {config}",
        "Model testing"
    )


def generate_forecast(init_date, config=1):
    """Generate forecast for a single initialization date."""
    run_command(
        f"./generate_forecast.sh {init_date} {config}",
        f"Generate forecast for {init_date}"
    )


def extract_maranhao(init_date):
    """Extract Maranhão region from forecast."""
    datadir = os.getenv('TELNET_DATADIR')
    results_dir = os.path.join(datadir, 'results', init_date)
    shapefile_dir = os.path.join(datadir, 'shapefiles')

    # Find forecast file
    forecast_files = [f for f in os.listdir(results_dir) if f.endswith('.nc')]

    for f in forecast_files:
        input_path = os.path.join(results_dir, f)
        output_path = os.path.join(results_dir, f'maranhao_{f}')

        run_command(
            f"python extract_maranhao.py {input_path} {output_path} --shapefile-dir {shapefile_dir}",
            f"Extract Maranhão from {f}"
        )


def verify_forecast(init_date):
    """Verify forecast against observations."""
    datadir = os.getenv('TELNET_DATADIR')
    results_dir = os.path.join(datadir, 'results', init_date)
    shapefile_dir = os.path.join(datadir, 'shapefiles')

    # Forecast and observation paths
    forecast_path = os.path.join(results_dir, 'maranhao_ensemble.nc')
    obs_path = os.path.join(datadir, 'era5', 'era5_pr_2025-2025_preprocessed.nc')
    output_dir = os.path.join(results_dir, 'verification')

    run_command(
        f"python verify_forecasts.py {forecast_path} {obs_path} -o {output_dir} --shapefile-dir {shapefile_dir}",
        f"Verify forecast {init_date}"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Run TelNet seasonal forecasts for Maranhão'
    )
    parser.add_argument(
        '--init-dates', nargs='+', default=['202501', '202504', '202507', '202510'],
        help='Initialization dates (YYYYMM format)'
    )
    parser.add_argument(
        '--skip-download', action='store_true',
        help='Skip data download step'
    )
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Skip feature selection and model training'
    )
    parser.add_argument(
        '--n-samples', type=int, default=100,
        help='Number of samples for feature selection and model selection'
    )
    parser.add_argument(
        '--config', type=int, default=1,
        help='Model configuration to use'
    )

    args = parser.parse_args()

    print("="*60)
    print("TelNet Maranhão Seasonal Forecast Workflow")
    print("="*60)
    print(f"Initialization dates: {args.init_dates}")
    print(f"Samples: {args.n_samples}")
    print(f"Config: {args.config}")

    # Setup
    setup_environment()

    # Data download
    if not args.skip_download:
        download_data()
        compute_indices()

    # Model training
    if not args.skip_training:
        run_feature_selection(args.n_samples)
        run_model_selection(args.n_samples)
        run_model_testing(args.n_samples, args.config)

    # Generate forecasts for each initialization date
    for init_date in args.init_dates:
        print(f"\n{'#'*60}")
        print(f"Processing initialization: {init_date}")
        print(f"{'#'*60}")

        generate_forecast(init_date, args.config)
        extract_maranhao(init_date)
        verify_forecast(init_date)

    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {os.path.join(os.getenv('TELNET_DATADIR'), 'results')}")


if __name__ == "__main__":
    main()
```

**Step 2: Verify Python syntax**

Run: `python -m py_compile run_maranhao_forecast.py`
Expected: No output (valid syntax)

**Step 3: Commit**

```bash
git add run_maranhao_forecast.py
git commit -m "feat: add main workflow orchestrator for Maranhão forecasts"
```

---

## Task 8: Create Colab Setup Instructions

**Files:**
- Create: `docs/COLAB_SSH_SETUP.md`

**Step 1: Create the setup documentation**

```markdown
# Google Colab SSH Setup

This guide explains how to run TelNet on Google Colab while working from your local terminal.

## Prerequisites

- Google account
- Google Colab access (free tier works, Pro recommended for longer sessions)
- SSH client on your local machine

## Step 1: Open the Setup Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Upload `colab_ssh_setup.ipynb` from this repository
3. Change runtime to GPU: Runtime → Change runtime type → T4 GPU

## Step 2: Run the Setup Cells

Run each cell in order:

1. **Cell 1**: Installs cloudflared
2. **Cell 2**: Sets root password and starts SSH (note the password!)
3. **Cell 3**: Starts the tunnel (copy the URL that appears)

## Step 3: Connect from Your Terminal

```bash
# Install cloudflared locally if not already installed
# macOS: brew install cloudflared
# Ubuntu: wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared-linux-amd64.deb

# Connect via SSH
ssh -o ProxyCommand="cloudflared access ssh --hostname %h" root@<tunnel-url>
```

Enter the password shown in Cell 2 when prompted.

## Step 4: Setup TelNet Environment

Once connected:

```bash
# Clone the repository
git clone https://github.com/<your-repo>/telnet.git
cd telnet

# Install dependencies
pip install -r docker/requirements.txt
pip install s3fs zarr geopandas rioxarray

# Set environment variables
export TELNET_DATADIR=/content/data
mkdir -p $TELNET_DATADIR

# Mount Google Drive for persistence (optional)
# Run this in a Colab cell first: from google.colab import drive; drive.mount('/content/drive')
# Then link: ln -s /content/drive/MyDrive/telnet_data $TELNET_DATADIR
```

## Step 5: Run the Workflow

```bash
# Full workflow
python run_maranhao_forecast.py --init-dates 202501 202504 202507 202510

# Or step by step:
python download_era5_aws.py -idate 194001 -fdate 202512
python download_maranhao_shapefile.py
python compute_climate_indices.py -fdate 202512
# ... etc
```

## Session Limits

| Tier | Max Session | Idle Timeout | GPU |
|------|-------------|--------------|-----|
| Free | ~12 hours | ~90 min | T4 |
| Pro | ~24 hours | ~90 min | T4/A100 |
| Pro+ | ~24 hours | ~90 min | A100 priority |

## Tips

- **Save progress frequently**: Push to git or sync to Drive
- **Use screen/tmux**: Keep processes running if SSH disconnects
- **Check GPU**: Run `nvidia-smi` to verify GPU is available
- **Resume training**: Scripts are designed to skip completed steps

## Troubleshooting

**Tunnel disconnects**: Re-run Cell 3 and reconnect

**Session dies**: Your files are lost. Always save to Drive or git.

**Out of memory**: Reduce batch size in model_selection.py
```

**Step 2: Commit**

```bash
git add docs/COLAB_SSH_SETUP.md
git commit -m "docs: add Colab SSH setup instructions"
```

---

## Task 9: Update Requirements

**Files:**
- Modify: `docker/requirements.txt`

**Step 1: Add new dependencies**

Add these lines to `docker/requirements.txt`:

```
s3fs>=2023.1.0
zarr>=2.14.0
geopandas>=0.12.0
rioxarray>=0.14.0
```

**Step 2: Commit**

```bash
git add docker/requirements.txt
git commit -m "deps: add AWS and geo dependencies for Maranhão workflow"
```

---

## Task 10: Final Integration Test

**Step 1: Verify all files exist**

Run:
```bash
ls -la colab_ssh_setup.ipynb download_era5_aws.py download_maranhao_shapefile.py extract_maranhao.py verify_forecasts.py run_maranhao_forecast.py configs/maranhao_bounds.txt docs/COLAB_SSH_SETUP.md
```

Expected: All files listed

**Step 2: Verify Python syntax for all new modules**

Run:
```bash
python -m py_compile download_era5_aws.py download_maranhao_shapefile.py extract_maranhao.py verify_forecasts.py run_maranhao_forecast.py
```

Expected: No output (all valid)

**Step 3: Final commit with all changes**

```bash
git log --oneline -10
```

Expected: See all commits from this implementation

---

## Summary

After completing all tasks, you will have:

| File | Purpose |
|------|---------|
| `colab_ssh_setup.ipynb` | Notebook to create SSH tunnel into Colab |
| `download_era5_aws.py` | ERA5 downloader using AWS (no CDS needed) |
| `download_maranhao_shapefile.py` | IBGE shapefile downloader |
| `extract_maranhao.py` | Clip/mask data to Maranhão boundaries |
| `verify_forecasts.py` | Skill scores and reliability diagrams |
| `run_maranhao_forecast.py` | Main workflow orchestrator |
| `configs/maranhao_bounds.txt` | Geographic bounds configuration |
| `docs/COLAB_SSH_SETUP.md` | Setup instructions |

To run the full workflow on Colab:
```bash
ssh into colab
cd telnet
export TELNET_DATADIR=/content/data
python run_maranhao_forecast.py --init-dates 202501 202504 202507 202510
```
