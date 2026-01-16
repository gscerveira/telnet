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
