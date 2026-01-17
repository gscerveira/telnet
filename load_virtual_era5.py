"""
Load ERA5 data from virtual Icechunk stores.
Streams data on-demand from S3 without local storage.
"""

import os
import xarray as xr
import icechunk


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
    valid_variables = ['precipitation', 'u10', 'v10', 'geopotential']
    if variable not in valid_variables:
        raise ValueError(f"Invalid variable '{variable}'. Must be one of: {valid_variables}")

    if datadir is None:
        datadir = os.getenv('TELNET_DATADIR', '/content/drive/MyDrive/telnet_data')

    store_path = os.path.join(datadir, 'virtual_stores', f'era5_{variable}')

    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"Virtual store not found: {store_path}\n"
            f"Run: python build_virtual_era5.py -idate 194001 -fdate 202512"
        )

    # Open existing Icechunk repository
    print(f"  [Icechunk] Opening repository: {store_path}")
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
    print(f"  [Icechunk] Connected to S3 bucket: {BUCKET} (streaming mode)")

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
        dims = list(ds.dims)
        if not dims:
            raise ValueError("Dataset has no dimensions")
        time_name = 'time' if 'time' in ds.dims else dims[0]
        data = data.sel({time_name: time_slice})

    # Get the data variable
    data_vars = list(data.data_vars)
    if not data_vars:
        raise ValueError("Dataset has no data variables")
    data_var = data_vars[0]
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
    data_vars = list(data.data_vars)
    if not data_vars:
        raise ValueError("Dataset has no data variables")
    data_var = data_vars[0]
    monthly = data[data_var].resample(time='ME').sum()

    # Select time range
    monthly = monthly.sel(time=slice(start_date, end_date))

    return monthly.load()
