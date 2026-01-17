from calendar import monthrange
import os
import pandas as pd
import wget
import argparse
import cdsapi
import xarray as xr
import numpy as np
from utils import exp_data_dir


def merge_present_data(var, final_date):

    root_datadir = os.getenv('TELNET_DATADIR')
    era5_dir = os.path.join(root_datadir, 'era5')
    old_file = os.path.join(era5_dir, f'era5_{var}_2025-present_preprocessed.nc')
    new_file = os.path.join(era5_dir, f'era5_{var}_2025-{final_date[:4]}_preprocessed.nc')
    if os.path.exists(old_file):
        ds_old = xr.open_dataset(old_file)
        ds_new = xr.open_dataset(new_file)
        ds_merged = xr.concat([ds_old, ds_new], dim='time')
        ds_merged.to_netcdf(old_file)
        os.remove(new_file)
        print (f'Merged {var} data for 2025-present with newly downloaded data up to {final_date}.')
    else:
        print (f'No existing 2025-present {var} data found to merge. Creating new file.')
        ds_new = xr.open_dataset(new_file)
        ds_new.to_netcdf(old_file)
        os.remove(new_file)

def update_dates(init_date, final_date):
    root_datadir = os.getenv('TELNET_DATADIR')
    era5_dir = os.path.join(root_datadir, 'era5')
    os.makedirs(era5_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(era5_dir) if f.startswith('era5_pr') and f.endswith('present_preprocessed.nc')]
    if len(existing_files) == 0:
        return init_date, final_date
    else:
        ds = xr.open_dataset(os.path.join(era5_dir, existing_files[0]))
        if np.datetime64(final_date) == ds['time'].values[-1]:
            print ('No new data to download. Exiting.')
            exit(0)
        init_date = np.datetime64((ds['time'].values[-1] + pd.DateOffset(months=1)))        
        return init_date.astype(str)[0:10], final_date

def download_ersstv5(init_date, final_date):

    root_datadir = os.getenv('TELNET_DATADIR')

    iyr = init_date[:4]
    fyr = final_date[:4]
    if int(fyr) > 2024:
        fyr = 'present'

    output_file = f"{root_datadir}/ersstv5_{iyr}-{fyr}.nc"

    if not os.path.exists(output_file):
        download_link = 'https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc'
        print(f"Downloading ERSSTv5 data for the period {init_date} to {final_date}...")
        wget.download(download_link, f"{root_datadir}/ersstv5_tmp.nc")
        ds = xr.open_dataset(f"{root_datadir}/ersstv5_tmp.nc")
        ds = ds.sel(time=slice(init_date, final_date)).drop('time_bnds')
        ds.to_netcdf(output_file)
        os.remove(f"{root_datadir}/ersstv5_tmp.nc")
        print(f"\nERSSTv5 data downloaded and saved to {output_file}")
    else:
        print(f"ERSSTv5 data already exists: {output_file}. Skipping download.")

def retrieve_monthly_era5(vars, years, months, output, levels=[]):

    c = cdsapi.Client()

    for var, out in zip(vars, output):
        if levels != []:
            c.retrieve(
                'reanalysis-era5-pressure-levels-monthly-means',
                {   'product_type': 'monthly_averaged_reanalysis',
                    'variable':  var,
                    'pressure_level': levels,
                    'year': years,
                    'month': months,
                    'time': '00:00',
                    'format': 'netcdf',
                },
                out)
        else:
            c.retrieve(
                "reanalysis-era5-single-levels-monthly-means",
                {
                    'product_type': 'monthly_averaged_reanalysis',
                    'variable': var,
                    'year': years,
                    'month': months,
                    'time': '00:00',
                    'format': 'netcdf',
                },
                out)

def download_era5(init_date, final_date):

    root_datadir = os.getenv('TELNET_DATADIR')
    era5_dir = os.path.join(root_datadir, 'era5')
    os.makedirs(era5_dir, exist_ok=True)
    iyr = init_date[:4]
    fyr = final_date[:4]

    PRESSURE_LEVEL_VARS = [
        "geopotential"
    ]
    levels = [500, 700, 1000]
    SINGLE_LEVEL_VARS = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_precipitation",
        "land_sea_mask",
    ]
    var_forms = {
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "total_precipitation": "pr",
        "geopotential": "hgt",
        "land_sea_mask": "land_sea_mask",
    }    

    yrs = [str(i) for i in range(int(iyr), int(fyr) + 1)]
    mons = [f'{i:02d}' for i in range(1, 13)]

    print (f"Downloading ERA5 data for the period {init_date} to {final_date}...")
    # Download pressure level data
    for var in PRESSURE_LEVEL_VARS:
        for level in levels:
            if not os.path.exists(os.path.join(era5_dir, f'era5_{var_forms[var]}_{level}_{iyr}-{fyr}.nc')):
                print (f"Downloading {var} at level {level}...")
                retrieve_monthly_era5([var], yrs, mons, [os.path.join(era5_dir, f'era5_{var}_{level}_tmp.nc')], levels=[level])
                ds = xr.open_dataset(os.path.join(era5_dir, f'era5_{var}_{level}_tmp.nc'))
                ds = ds.sel(valid_time=slice(init_date, final_date))
                ds.to_netcdf(os.path.join(era5_dir, f'era5_{var_forms[var]}_{level}_{iyr}-{fyr}.nc'))
                os.remove(os.path.join(era5_dir, f'era5_{var}_{level}_tmp.nc'))
            else:
                print (f"{var} data already exists. Skipping download.")
    # Download single level data
    for var in SINGLE_LEVEL_VARS:
        if not os.path.exists(os.path.join(era5_dir, f'era5_{var_forms[var]}_{iyr}-{fyr}.nc')):
            print (f"Downloading {var} ...")
            retrieve_monthly_era5([var], yrs, mons, [os.path.join(era5_dir, f'era5_{var}_tmp.nc')])
            ds = xr.open_dataset(os.path.join(era5_dir, f'era5_{var}_tmp.nc'))
            ds = ds.sel(valid_time=slice(init_date, final_date))
            ds.to_netcdf(os.path.join(era5_dir, f'era5_{var_forms[var]}_{iyr}-{fyr}.nc'))
            os.remove(os.path.join(era5_dir, f'era5_{var}_tmp.nc'))
        else:
            print (f"{var} data already exists. Skipping download.")

def preprocess_era5(var, init_date, final_date, lats, lons):

    """
    U10, V10: interpolate to 2dg, rename latitude to lat and longitude to lon
    HGT: interpolate to 2dg, convert to geopotential height in meters, rename height variable, rename valid_time to time, latitude to lat and longitude to lon
    PCP: convert to mm, cut to target_region, and rename tp to pr, latitude to lat and longitude to lon, shift longitude to -180 to 180, shift time by -6 hours
    Land-sea mask: cut to target_region, rename valid_time to time, latitude to lat and longitude to lon, shift longitude to -180 to 180
    """

    era5_dir = os.path.join(os.getenv('TELNET_DATADIR'), 'era5')
    iyr = init_date[:4]
    fyr = final_date[:4]
    if not os.path.exists(os.path.join(era5_dir, f'era5_{var}_{iyr}-{fyr}_preprocessed.nc')):
        lon2interp = np.arange(0., 360., 2.0)
        lat2interp = np.arange(-88., 90., 2.0)[::-1]

        ds = xr.open_dataset(os.path.join(era5_dir, f'era5_{var}_{iyr}-{fyr}.nc'))
        if var in ['u10', 'v10', 'hgt_500', 'hgt_700', 'hgt_1000']:
            ds = ds.interp(latitude=lat2interp, longitude=lon2interp, method='linear')
            ds = ds.rename({'valid_time': 'time','latitude': 'lat', 'longitude': 'lon'})
            if var in ['hgt_500', 'hgt_700', 'hgt_1000']:
                ds['z'] = ds['z'] / 9.80665
                ds = ds.rename({'z': 'height', 'pressure_level': 'level'})
        elif var == 'pr':
            ds = ds.rename({'tp': 'pr', 'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'})
            ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
            ds = ds.sel(lat=slice(lats[1], lats[0]), lon=slice(lons[0], lons[1]))
            ds['time'] = ds['time'] - np.timedelta64(6, 'h')
            ndays = np.tile(np.array([monthrange(i.year, i.month)[1] for i in pd.to_datetime(ds['pr'].time.values)])[:, None, None], ((1, ds['pr'].shape[1], ds['pr'].shape[2])))
            ds['pr'] = ds['pr'] * 1000 * ndays  # convert to mm
            ds['pr']['units'] = 'mm'
        elif var == 'land_sea_mask':
            ds = ds.rename({'valid_time': 'time','latitude': 'lat', 'longitude': 'lon'})
            ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
            ds = ds.sel(lat=slice(lats[1], lats[0]), lon=slice(lons[0], lons[1]))

        ds.to_netcdf(os.path.join(era5_dir, f'era5_{var}_{iyr}-{fyr}_preprocessed.nc'))
        os.remove(os.path.join(era5_dir, f'era5_{var}_{iyr}-{fyr}.nc'))
    else:
        print (f'Preprocessed {var} data already exists. Skipping preprocessing.')

def concatenate_level_datasets(var, init_date, final_date):

    era5_dir = os.path.join(os.getenv('TELNET_DATADIR'), 'era5')
    iyr = init_date[:4]
    fyr = final_date[:4]
    if not os.path.exists(os.path.join(era5_dir, f'era5_{var}_{iyr}-{fyr}_preprocessed.nc')):
        files = [os.path.join(era5_dir, f) for f in os.listdir(era5_dir) if f.startswith(f'era5_{var}') and f.endswith(f'_{iyr}-{fyr}_preprocessed.nc')]
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='level')
        ds = ds.sortby('level')
        ds.to_netcdf(os.path.join(era5_dir, f'era5_{var}_{iyr}-{fyr}_preprocessed.nc'))
        for f in files:
            os.remove(f)
    else:
        print (f'Concatenated {var} data already exists. Skipping concatenation.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download ERSST and ERA5 monthly data for the selected time period')
    parser.add_argument('-idate','--initdate', help='Initial download date in the format YYYYMM. Ex. 199202', required=True, default=None)
    parser.add_argument('-fdate','--finaldate', help='Final download date in the format YYYYMM. Ex. 199310', required=True, default=None)
    if not os.path.exists(os.path.join(exp_data_dir, 'lat_lon_boundaries.txt')):
        init_lat = input("Enter initial (southmost) latitude of the region (e.g., -15): ")
        final_lat = input("Enter final (northmost) latitude of the region (e.g., 10): ")
        init_lon = input("Enter initial (westmost) longitude of the region (e.g., -85): ")
        final_lon = input("Enter final (eastmost) longitude of the region (e.g., -30): ")
        with open(os.path.join(exp_data_dir, 'lat_lon_boundaries.txt'), 'w') as f:
            f.write(f"{init_lat}\n{final_lat}\n{init_lon}\n{final_lon}\n")
    else:
        with open(os.path.join(exp_data_dir, 'lat_lon_boundaries.txt'), 'r') as f:
            lines = f.readlines()
            init_lat = lines[0].strip()
            final_lat = lines[1].strip()
            init_lon = lines[2].strip()
            final_lon = lines[3].strip()

    args = vars(parser.parse_args())
    init_date = f'{args["initdate"][:4]}-{args["initdate"][4:6]}-01'
    final_date = f'{args["finaldate"][:4]}-{args["finaldate"][4:6]}-01'
    lats = [float(init_lat), float(final_lat)]
    lons = [float(init_lon), float(final_lon)]

    if int(final_date[:4]) > 2024:
        init_date, final_date = update_dates(init_date, final_date)

    if not os.path.exists(os.path.join(os.getenv('TELNET_DATADIR'), f'ersstv5_{init_date[:4]}-{final_date[:4]}.nc')):
        download_ersstv5(init_date, final_date)
    else:
        print (f"ERSSTv5 data for the period {init_date} to {final_date} already exists. Skipping download.")

    final_vars = ['u10', 'v10', 'pr', 'hgt', 'land_sea_mask']
    if not all([os.path.exists(os.path.join(os.getenv('TELNET_DATADIR'), 'era5', f'era5_{var}_{init_date[:4]}-{final_date[:4]}_preprocessed.nc')) for var in final_vars]):
        download_era5(init_date, final_date)
        vars = ['u10', 'v10', 'pr', 'hgt_500', 'hgt_700', 'hgt_1000', 'land_sea_mask']
        for var in vars:
            preprocess_era5(var, init_date, final_date, lats, lons)
        concatenate_level_datasets('hgt', init_date, final_date)
        if int(final_date[:4]) > 2024:
            # Concatenate existing 2025-present data with downloaded data
            for var in final_vars:
                merge_present_data(var, final_date)

    else:
        print ('Preprocessed ERA5 data for all variables already exists. Skipping download and preprocessing.')
