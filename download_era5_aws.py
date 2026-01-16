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
