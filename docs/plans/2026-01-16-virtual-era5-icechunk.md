# Virtual ERA5 with Icechunk Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace ERA5 downloads with virtual Zarr references using Icechunk + VirtualiZarr, eliminating local storage and enabling on-demand streaming.

**Architecture:** Create a virtual dataset that references the NSF NCAR ERA5 NetCDF files directly on S3. The Icechunk repository stores only metadata (~MB) while data is fetched on-demand from the source. Regional subsetting happens at read time via xarray's lazy loading.

**Tech Stack:** icechunk, virtualizarr, xarray, zarr (v3), s3fs

---

## Task 1: Add Icechunk Dependencies

**Files:**
- Modify: `docker/requirements.txt`

**Step 1: Add new dependencies**

Add to `docker/requirements.txt`:
```
icechunk>=1.0.0
virtualizarr>=1.0.0
zarr>=3.0.0
```

**Step 2: Update Colab notebook cell 2**

In `colab_maranhao_workflow.ipynb`, update the install command:
```python
!uv pip install --system -q -r docker/requirements.txt
!uv pip install --system -q s3fs geopandas rioxarray icechunk virtualizarr
```

**Step 3: Commit**

```bash
git add docker/requirements.txt colab_maranhao_workflow.ipynb
git commit -m "deps: add icechunk and virtualizarr for virtual datasets"
```

---

## Task 2: Create Virtual ERA5 Store Builder

**Files:**
- Create: `build_virtual_era5.py`

**Step 1: Create the virtual store builder**

```python
"""
Build a virtual Zarr store for ERA5 data using Icechunk + VirtualiZarr.
References NetCDF files on S3 without downloading them.
"""

import os
import argparse
import fsspec
import xarray as xr
import icechunk
from virtualizarr import open_virtual_dataset
from utils import exp_data_dir


BUCKET = "nsf-ncar-era5"
S3_PREFIX = f"s3://{BUCKET}/"


def get_precipitation_files(fs, start_year, end_year):
    """Get list of precipitation files for date range."""
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yyyymm = f"{year}{month:02d}"
            # Large-scale precipitation
            lsp = fs.glob(f"{BUCKET}/e5.oper.fc.sfc.accumu/{yyyymm}/e5.oper.fc.sfc.accumu.128_142_lsp.ll025sc.*.nc")
            # Convective precipitation
            cp = fs.glob(f"{BUCKET}/e5.oper.fc.sfc.accumu/{yyyymm}/e5.oper.fc.sfc.accumu.128_143_cp.ll025sc.*.nc")
            files.extend([f"s3://{f}" for f in lsp + cp])
    return sorted(files)


def get_wind_files(fs, start_year, end_year, component='10u'):
    """Get list of wind files for date range."""
    var_code = '128_165_10u' if component == '10u' else '128_166_10v'
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yyyymm = f"{year}{month:02d}"
            matches = fs.glob(f"{BUCKET}/e5.oper.an.sfc/{yyyymm}/e5.oper.an.sfc.{var_code}.ll025sc.*.nc")
            files.extend([f"s3://{f}" for f in matches])
    return sorted(files)


def get_geopotential_files(fs, start_year, end_year):
    """Get list of geopotential files for date range."""
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yyyymm = f"{year}{month:02d}"
            matches = fs.glob(f"{BUCKET}/e5.oper.an.pl/{yyyymm}/e5.oper.an.pl.128_129_z.ll025sc.*.nc")
            files.extend([f"s3://{f}" for f in matches])
    return sorted(files)


def build_virtual_store(files, store_path, variable_name):
    """
    Build a virtual Icechunk store from a list of NetCDF files.
    """
    print(f"Building virtual store for {variable_name} with {len(files)} files...")

    # Create virtual datasets from each file
    virtual_datasets = []
    for i, url in enumerate(files):
        if i % 100 == 0:
            print(f"  Processing file {i+1}/{len(files)}...")
        try:
            vds = open_virtual_dataset(url, indexes={})
            virtual_datasets.append(vds)
        except Exception as e:
            print(f"  Warning: Could not virtualize {url}: {e}")
            continue

    if not virtual_datasets:
        print(f"  No files virtualized for {variable_name}")
        return None

    # Combine into single virtual dataset
    print(f"  Combining {len(virtual_datasets)} virtual datasets...")
    virtual_ds = xr.concat(
        virtual_datasets,
        dim='time',
        coords='minimal',
        compat='override',
        combine_attrs='override'
    )

    # Create Icechunk repository
    storage = icechunk.local_filesystem_storage(path=store_path)
    config = icechunk.RepositoryConfig.default()

    # Configure virtual chunk container for S3
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            S3_PREFIX,
            icechunk.s3_store(region="us-west-2")
        )
    )

    # Set up anonymous S3 credentials
    credentials = icechunk.containers_credentials({
        S3_PREFIX: icechunk.s3_credentials(anonymous=True)
    })

    repo = icechunk.Repository.create(storage, config, credentials)

    # Write virtual dataset to Icechunk
    session = repo.writable_session("main")
    virtual_ds.virtualize.to_icechunk(session.store)
    snapshot_id = session.commit(f"Initial {variable_name} virtual store")

    print(f"  Created virtual store at {store_path} (snapshot: {snapshot_id})")
    return store_path


def main():
    parser = argparse.ArgumentParser(
        description='Build virtual ERA5 Icechunk stores (no data download)'
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
    parser.add_argument(
        '--variable',
        choices=['precipitation', 'u10', 'v10', 'geopotential', 'all'],
        default='all',
        help='Which variable to build (default: all)'
    )

    args = parser.parse_args()

    start_year = int(args.initdate[:4])
    end_year = int(args.finaldate[:4])

    root_datadir = os.getenv('TELNET_DATADIR', '/content/drive/MyDrive/telnet_data')
    virtual_dir = os.path.join(root_datadir, 'virtual_stores')
    os.makedirs(virtual_dir, exist_ok=True)

    fs = fsspec.filesystem('s3', anon=True)

    variables = [args.variable] if args.variable != 'all' else ['precipitation', 'u10', 'v10', 'geopotential']

    for var in variables:
        store_path = os.path.join(virtual_dir, f'era5_{var}')

        if os.path.exists(store_path):
            print(f"Virtual store already exists: {store_path}")
            continue

        if var == 'precipitation':
            files = get_precipitation_files(fs, start_year, end_year)
        elif var == 'u10':
            files = get_wind_files(fs, start_year, end_year, '10u')
        elif var == 'v10':
            files = get_wind_files(fs, start_year, end_year, '10v')
        elif var == 'geopotential':
            files = get_geopotential_files(fs, start_year, end_year)

        build_virtual_store(files, store_path, var)

    print("\nVirtual ERA5 stores complete!")
    print(f"Stores saved to: {virtual_dir}")
    print("No data was downloaded - only metadata references were created.")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add build_virtual_era5.py
git commit -m "feat: add virtual ERA5 store builder with Icechunk"
```

---

## Task 3: Create Virtual ERA5 Reader Module

**Files:**
- Create: `load_virtual_era5.py`

**Step 1: Create the reader module**

```python
"""
Load ERA5 data from virtual Icechunk stores.
Streams data on-demand from S3 without local storage.
"""

import os
import xarray as xr
import icechunk
import numpy as np


BUCKET = "nsf-ncar-era5"
S3_PREFIX = f"s3://{BUCKET}/"


def open_virtual_era5(variable, datadir=None):
    """
    Open a virtual ERA5 store for the specified variable.

    Parameters
    ----------
    variable : str
        One of: 'precipitation', 'u10', 'v10', 'geopotential'
    datadir : str, optional
        Path to telnet data directory. Uses TELNET_DATADIR env var if not provided.

    Returns
    -------
    xarray.Dataset
        Lazy-loaded dataset that streams from S3 on demand.
    """
    if datadir is None:
        datadir = os.getenv('TELNET_DATADIR', '/content/drive/MyDrive/telnet_data')

    store_path = os.path.join(datadir, 'virtual_stores', f'era5_{variable}')

    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"Virtual store not found: {store_path}\n"
            f"Run: python build_virtual_era5.py -idate 194001 -fdate 202512"
        )

    # Open existing Icechunk repository
    storage = icechunk.local_filesystem_storage(path=store_path)
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            S3_PREFIX,
            icechunk.s3_store(region="us-west-2")
        )
    )
    credentials = icechunk.containers_credentials({
        S3_PREFIX: icechunk.s3_credentials(anonymous=True)
    })

    repo = icechunk.Repository.open(storage, config, credentials)
    session = repo.readonly_session(branch="main")

    ds = xr.open_zarr(session.store, zarr_version=3, consolidated=False, chunks={})
    return ds


def load_era5_region(variable, lats, lons, time_slice=None, datadir=None):
    """
    Load ERA5 data for a specific region, streaming from S3.

    Parameters
    ----------
    variable : str
        One of: 'precipitation', 'u10', 'v10', 'geopotential'
    lats : tuple
        (min_lat, max_lat) for region
    lons : tuple
        (min_lon, max_lon) for region
    time_slice : slice, optional
        Time slice to load (e.g., slice('2020-01', '2020-12'))
    datadir : str, optional
        Path to telnet data directory

    Returns
    -------
    xarray.DataArray
        Data for the specified region, loaded into memory.
    """
    ds = open_virtual_era5(variable, datadir)

    # Determine coordinate names
    lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'

    # Handle longitude conversion if needed (0-360 vs -180-180)
    lon_values = ds[lon_name].values
    if lon_values.max() > 180 and lons[0] < 0:
        # Convert request to 0-360
        lons = (lons[0] + 360, lons[1] + 360)

    # Subset to region
    data = ds.sel({
        lat_name: slice(lats[1], lats[0]),
        lon_name: slice(lons[0], lons[1])
    })

    if time_slice is not None:
        time_name = 'time' if 'time' in ds.dims else list(ds.dims)[0]
        data = data.sel({time_name: time_slice})

    # Get the data variable
    data_var = list(data.data_vars)[0]
    return data[data_var].load()


def compute_monthly_precip(lats, lons, start_date, end_date, datadir=None):
    """
    Compute monthly total precipitation for a region.

    This replaces download_era5_precipitation_aws() but streams data
    instead of downloading it.
    """
    ds = open_virtual_era5('precipitation', datadir)

    lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'

    # Subset to region with buffer
    lat_buffer = 5
    lon_buffer = 5

    data = ds.sel({
        lat_name: slice(lats[1] + lat_buffer, lats[0] - lat_buffer),
        lon_name: slice(lons[0] - lon_buffer, lons[1] + lon_buffer)
    })

    # Get data variable and resample to monthly
    data_var = list(data.data_vars)[0]
    monthly = data[data_var].resample(time='ME').sum()

    # Select time range
    monthly = monthly.sel(time=slice(start_date, end_date))

    return monthly.load()
```

**Step 2: Commit**

```bash
git add load_virtual_era5.py
git commit -m "feat: add virtual ERA5 reader module for on-demand streaming"
```

---

## Task 4: Update Colab Workflow Notebook

**Files:**
- Modify: `colab_maranhao_workflow.ipynb`

**Step 1: Add new cell for building virtual stores (Cell 4 alternative)**

Add a new cell after dependencies install:

```python
# Cell 4 - Build Virtual ERA5 Stores (FAST - only creates references)
# This replaces the slow download step - no data is actually downloaded!
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'

# Build virtual stores (~5-10 minutes to index files, NO download)
!python build_virtual_era5.py -idate 194001 -fdate 202512

print("Virtual stores created! Data will stream on-demand from S3.")
```

**Step 2: Add cell showing how to use virtual data**

```python
# Cell 4b - Verify virtual stores work
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'

from load_virtual_era5 import open_virtual_era5, load_era5_region

# Open the virtual precipitation store
ds = open_virtual_era5('precipitation')
print("Virtual dataset:")
print(ds)

# Load just Maranhão region for January 2020 (streams only needed data)
lats = (-10.25, -1.0)
lons = (-48.75, -41.5)
data = load_era5_region('precipitation', lats, lons, time_slice=slice('2020-01', '2020-01'))
print(f"\nLoaded Maranhão data shape: {data.shape}")
print(f"Data size in memory: {data.nbytes / 1e6:.2f} MB")
```

**Step 3: Commit**

```bash
git add colab_maranhao_workflow.ipynb
git commit -m "feat: add virtual ERA5 workflow cells to Colab notebook"
```

---

## Task 5: Update Main Workflow to Use Virtual Data

**Files:**
- Modify: `run_maranhao_forecast.py`

**Step 1: Add option to use virtual stores**

Add a `--virtual` flag that uses the virtual ERA5 data instead of downloading:

```python
parser.add_argument(
    '--virtual',
    action='store_true',
    help='Use virtual Icechunk stores instead of downloading data'
)
```

**Step 2: Update data loading logic**

In the data preparation section, check for virtual mode:

```python
if args.virtual:
    print("Using virtual ERA5 stores (streaming from S3)...")
    # Build virtual stores if they don't exist
    virtual_dir = os.path.join(datadir, 'virtual_stores')
    if not os.path.exists(virtual_dir):
        subprocess.run([
            'python', 'build_virtual_era5.py',
            '-idate', '194001', '-fdate', final_date.replace('-', '')[:6]
        ], check=True)
else:
    # Original download logic
    subprocess.run([...])
```

**Step 3: Commit**

```bash
git add run_maranhao_forecast.py
git commit -m "feat: add --virtual flag to use Icechunk stores"
```

---

## Summary

| Before (Download) | After (Virtual) |
|-------------------|-----------------|
| Downloads ~5-20 GB | Stores ~50 MB metadata |
| Takes 30-60 min | Takes 5-10 min to index |
| Data on local disk | Data streams from S3 |
| Must re-download for updates | Always up-to-date |

**Usage:**

```bash
# Build virtual stores (once)
python build_virtual_era5.py -idate 194001 -fdate 202512

# Run forecast with virtual data
python run_maranhao_forecast.py --virtual --init-dates 202501 202504
```

**Sources:**
- [Icechunk Virtual Datasets](https://icechunk.io/en/latest/virtual/)
- [VirtualiZarr Documentation](https://virtualizarr.readthedocs.io/en/stable/index.html)
- [Icechunk FAQ](https://icechunk.io/en/stable/faq/)
