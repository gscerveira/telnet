"""Load ERA5 data directly from ARCO ERA5 Zarr on GCS (no virtualization)."""
import xarray as xr
import gcsfs

ARCO_ERA5_PATH = 'gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

# Map short names to ARCO ERA5 variable names
VARIABLE_MAP = {
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'geopotential': 'geopotential',
    'height': 'geopotential',  # will be converted
}


def open_arco_era5(variable='total_precipitation', period=None):
    """
    Open ERA5 from ARCO ERA5 Zarr store on GCS.

    Variables: 'total_precipitation', 'u10', 'v10', 'height', 'geopotential', etc.

    Args:
        variable: Variable name to load. Special handling for:
                  - 'total_precipitation': convective + large_scale
                  - 'height': geopotential / 9.80665
                  - 'u10', 'v10': mapped to ARCO names
        period: Optional tuple of (start_date, end_date) strings to slice time.

    Returns:
        xarray.Dataset with the requested variable.
    """
    fs = gcsfs.GCSFileSystem(token='anon')
    store = fs.get_mapper(ARCO_ERA5_PATH)
    ds = xr.open_zarr(store, consolidated=True)

    if variable == 'total_precipitation':
        # Combine convective + large-scale
        da = ds['convective_precipitation'] + ds['large_scale_precipitation']
        ds = da.to_dataset(name='total_precipitation')
    elif variable == 'height':
        # Convert geopotential to geopotential height
        da = ds['geopotential'] / 9.80665
        ds = da.to_dataset(name='height')
    elif variable in VARIABLE_MAP:
        arco_name = VARIABLE_MAP[variable]
        da = ds[arco_name]
        ds = da.to_dataset(name=variable)
    else:
        ds = ds[[variable]]

    if period:
        ds = ds.sel(time=slice(period[0], period[1]))

    return ds
