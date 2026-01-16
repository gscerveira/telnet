"""
Download ERA5 data from AWS Open Data Registry (NSF NCAR curated dataset).
Uses s3://nsf-ncar-era5 which requires no authentication and has data from 1940-2025.
"""

import os
import argparse
import numpy as np
import xarray as xr
import s3fs
from utils import exp_data_dir


# NSF NCAR ERA5 bucket
BUCKET = "nsf-ncar-era5"


def get_era5_aws_client():
    """Create S3 filesystem client for anonymous access."""
    return s3fs.S3FileSystem(anon=True)


def list_s3_files(fs, prefix):
    """List files matching a prefix in S3."""
    try:
        return fs.glob(f"s3://{BUCKET}/{prefix}")
    except Exception as e:
        print(f"Warning: Could not list {prefix}: {e}")
        return []


def download_era5_precipitation_aws(init_date, final_date, lats, lons):
    """
    Download ERA5 monthly precipitation from AWS Open Data.

    Downloads large-scale and convective precipitation, sums them for total.
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

            yyyymm = f"{year}{month:02d}"

            # Find precipitation files for this month
            lsp_pattern = f"e5.oper.fc.sfc.accumu/{yyyymm}/e5.oper.fc.sfc.accumu.128_142_lsp.ll025sc.*.nc"
            cp_pattern = f"e5.oper.fc.sfc.accumu/{yyyymm}/e5.oper.fc.sfc.accumu.128_143_cp.ll025sc.*.nc"

            lsp_files = list_s3_files(fs, lsp_pattern)
            cp_files = list_s3_files(fs, cp_pattern)

            if not lsp_files or not cp_files:
                print(f"Warning: No precipitation data for {year}-{month:02d}")
                continue

            try:
                print(f"Downloading precipitation for {year}-{month:02d}...")

                # Open and combine all files for this month
                lsp_data = xr.open_mfdataset(
                    [fs.open(f) for f in sorted(lsp_files)],
                    combine='by_coords',
                    engine='h5netcdf'
                )
                cp_data = xr.open_mfdataset(
                    [fs.open(f) for f in sorted(cp_files)],
                    combine='by_coords',
                    engine='h5netcdf'
                )

                # Get variable names (may vary)
                lsp_var = list(lsp_data.data_vars)[0]
                cp_var = list(cp_data.data_vars)[0]

                # Sum large-scale and convective precipitation
                total_precip = lsp_data[lsp_var] + cp_data[cp_var]

                # Compute monthly total (sum over all time dimensions)
                # ERA5 forecast data uses 'forecast_initial_time' and 'forecast_hour' instead of 'time'
                time_dims = [d for d in total_precip.dims if d not in ('latitude', 'longitude')]
                monthly_total = total_precip.sum(dim=time_dims)

                # Subset to region with buffer
                lat_buffer = 5
                lon_buffer = 5
                monthly_total = monthly_total.sel(
                    latitude=slice(lats[1] + lat_buffer, lats[0] - lat_buffer),
                    longitude=slice(lons[0] + 360 - lon_buffer if lons[0] < 0 else lons[0] - lon_buffer,
                                   lons[1] + 360 + lon_buffer if lons[1] < 0 else lons[1] + lon_buffer)
                )

                # Add time dimension
                monthly_total = monthly_total.expand_dims(
                    time=[np.datetime64(f'{year}-{month:02d}-01')]
                )
                monthly_data.append(monthly_total)

                lsp_data.close()
                cp_data.close()

            except Exception as e:
                print(f"Warning: Could not process {year}-{month:02d}: {e}")
                continue

    if monthly_data:
        combined = xr.concat(monthly_data, dim='time')
        combined = combined.rename('pr')

        # Rename coordinates to standard names
        if 'latitude' in combined.dims:
            combined = combined.rename({'latitude': 'lat', 'longitude': 'lon'})

        # Convert longitude from 0-360 to -180-180 if needed
        if combined.lon.values.max() > 180:
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

    for var_name, var_code in [('u10', '128_165_10u'), ('v10', '128_166_10v')]:
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

                yyyymm = f"{year}{month:02d}"
                pattern = f"e5.oper.an.sfc/{yyyymm}/e5.oper.an.sfc.{var_code}.ll025sc.*.nc"
                files = list_s3_files(fs, pattern)

                if not files:
                    print(f"Warning: No {var_name} data for {year}-{month:02d}")
                    continue

                try:
                    print(f"Downloading {var_name} for {year}-{month:02d}...")

                    ds = xr.open_mfdataset(
                        [fs.open(f) for f in sorted(files)],
                        combine='by_coords',
                        engine='h5netcdf'
                    )

                    data_var = list(ds.data_vars)[0]
                    # Find time dimension(s) dynamically
                    time_dims = [d for d in ds[data_var].dims if d not in ('latitude', 'longitude', 'level')]
                    monthly_mean = ds[data_var].mean(dim=time_dims)
                    monthly_mean = monthly_mean.expand_dims(
                        time=[np.datetime64(f'{year}-{month:02d}-01')]
                    )
                    monthly_data.append(monthly_mean)
                    ds.close()

                except Exception as e:
                    print(f"Warning: Could not load {year}-{month:02d}: {e}")
                    continue

        if monthly_data:
            combined = xr.concat(monthly_data, dim='time')
            combined = combined.rename(var_name)

            if 'latitude' in combined.dims:
                combined = combined.rename({'latitude': 'lat', 'longitude': 'lon'})

            # Interpolate to 2-degree grid for global teleconnections
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
    monthly_data = []

    for year in range(iyr, fyr + 1):
        for month in range(1, 13):
            if year == iyr and month < int(init_date[5:7]):
                continue
            if year == fyr and month > int(final_date[5:7]):
                continue

            yyyymm = f"{year}{month:02d}"
            # Geopotential is stored as daily files
            pattern = f"e5.oper.an.pl/{yyyymm}/e5.oper.an.pl.128_129_z.ll025sc.*.nc"
            files = list_s3_files(fs, pattern)

            if not files:
                print(f"Warning: No geopotential data for {year}-{month:02d}")
                continue

            try:
                print(f"Downloading geopotential for {year}-{month:02d}...")

                ds = xr.open_mfdataset(
                    [fs.open(f) for f in sorted(files)],
                    combine='by_coords',
                    engine='h5netcdf'
                )

                data_var = list(ds.data_vars)[0]

                # Select pressure levels 500, 700, 1000 hPa
                levels = [500, 700, 1000]
                if 'level' in ds.dims:
                    ds = ds.sel(level=levels, method='nearest')

                # Convert geopotential to geopotential height
                # Find time dimension(s) dynamically
                time_dims = [d for d in ds[data_var].dims if d not in ('latitude', 'longitude', 'level')]
                monthly_mean = ds[data_var].mean(dim=time_dims) / 9.80665
                monthly_mean = monthly_mean.expand_dims(
                    time=[np.datetime64(f'{year}-{month:02d}-01')]
                )
                monthly_data.append(monthly_mean)
                ds.close()

            except Exception as e:
                print(f"Warning: Could not load {year}-{month:02d}: {e}")
                continue

    if monthly_data:
        combined = xr.concat(monthly_data, dim='time')
        combined = combined.rename('height')

        if 'latitude' in combined.dims:
            combined = combined.rename({'latitude': 'lat', 'longitude': 'lon'})

        # Interpolate to 2-degree grid
        lon2interp = np.arange(0., 360., 2.0)
        lat2interp = np.arange(-88., 90., 2.0)[::-1]
        combined = combined.interp(lat=lat2interp, lon=lon2interp, method='linear')

        combined.to_netcdf(output_file)
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

    # Land-sea mask is in the invariant folder
    mask_path = f"s3://{BUCKET}/e5.oper.invariant/197901/e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc"

    try:
        print("Downloading land-sea mask...")
        ds = xr.open_dataset(fs.open(mask_path), engine='h5netcdf')

        data_var = list(ds.data_vars)[0]
        mask = ds[data_var].squeeze()

        if 'latitude' in mask.dims:
            mask = mask.rename({'latitude': 'lat', 'longitude': 'lon'})

        # Convert longitude from 0-360 to -180-180 if needed
        if mask.lon.values.max() > 180:
            mask = mask.assign_coords(
                lon=(((mask.lon + 180) % 360) - 180)
            ).sortby('lon')

        # Subset to region
        mask = mask.sel(
            lat=slice(lats[1], lats[0]),
            lon=slice(lons[0], lons[1])
        )

        mask = mask.expand_dims(time=[np.datetime64('1979-01-01')])
        mask.to_netcdf(output_file)
        print(f"Saved land-sea mask to {output_file}")
        ds.close()

    except Exception as e:
        print(f"Error downloading land-sea mask: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Download ERA5 data from AWS Open Data (NSF NCAR) for TelNet'
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
    print(f"Using NSF NCAR ERA5 bucket: s3://{BUCKET}/")

    # Download all variables
    download_era5_precipitation_aws(init_date, final_date, lats, lons)
    download_era5_winds_aws(init_date, final_date)
    download_era5_geopotential_aws(init_date, final_date)
    download_era5_land_sea_mask_aws(lats, lons)

    print("\nERA5 download complete!")


if __name__ == "__main__":
    main()
