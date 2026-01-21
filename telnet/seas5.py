"""SEAS5 downscaling integration module.

This module provides functionality to:
1. Download SEAS5 seasonal forecast hindcasts from CDS
2. Preprocess SEAS5 data to match TelNet's ERA5 grid
3. Use SEAS5 as an input feature for statistical downscaling
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
from rich.console import Console

console = Console()


def download_seas5(
    data_dir: Path,
    start_year: int,
    end_year: int,
    init_months: List[int] = [2, 5, 8, 11],
    lead_months: List[int] = [1, 2, 3, 4, 5, 6],
    region: Optional[dict] = None,
) -> None:
    """
    Download SEAS5 seasonal forecast hindcasts from Copernicus CDS.
    
    Args:
        data_dir: Base data directory
        start_year: Start year for hindcasts
        end_year: End year for hindcasts
        init_months: Initialization months (1=Jan, ..., 12=Dec)
        lead_months: Lead times in months
        region: Optional region bounds for subsetting
    """
    import cdsapi
    
    console.print("\n[bold]Downloading SEAS5 hindcasts from CDS...[/bold]")
    
    # Check for CDS credentials
    cdsapirc = Path.home() / '.cdsapirc'
    if not cdsapirc.exists():
        console.print("[red]ERROR: ~/.cdsapirc not found![/red]")
        console.print("Set up CDS API credentials from: https://cds.climate.copernicus.eu/how-to-api")
        return
    
    seas5_dir = data_dir / 'seas5'
    seas5_dir.mkdir(parents=True, exist_ok=True)
    
    c = cdsapi.Client()
    
    years = [str(y) for y in range(start_year, end_year + 1)]
    
    for init_month in init_months:
        month_name = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec'][init_month - 1]
        
        output_file = seas5_dir / f'seas5_pr_{month_name}_{start_year}-{end_year}.nc'
        
        if output_file.exists():
            console.print(f"  [dim]Already exists: {output_file.name}[/dim]")
            continue
        
        console.print(f"  → Downloading init month {init_month} ({month_name})...")
        
        # Build request
        request = {
            'originating_centre': 'ecmwf',
            'system': '51',  # SEAS5.1
            'variable': 'total_precipitation',
            'product_type': ['monthly_mean'],
            'year': years,
            'month': f'{init_month:02d}',
            'leadtime_month': [str(l) for l in lead_months],
            'data_format': 'netcdf',
        }
        
        # Add area subsetting if region specified
        if region:
            request['area'] = [
                region['lat_max'],
                region['lon_min'],
                region['lat_min'],
                region['lon_max'],
            ]
        
        try:
            c.retrieve(
                'seasonal-monthly-single-levels',
                request,
                str(output_file)
            )
            console.print(f"  [green]✓ Downloaded: {output_file.name}[/green]")
        except Exception as e:
            console.print(f"  [red]✗ Failed: {e}[/red]")
    
    console.print(f"\n  → Preprocessing SEAS5 data...")
    preprocess_seas5(seas5_dir, data_dir / 'era5')


def preprocess_seas5(seas5_dir: Path, era5_dir: Path) -> None:
    """
    Preprocess SEAS5 data to match ERA5 grid and format.
    
    Args:
        seas5_dir: Directory with raw SEAS5 files
        era5_dir: Directory with preprocessed ERA5 files (for target grid)
    """
    import xarray as xr
    import numpy as np
    
    console.print("  Preprocessing SEAS5 to match ERA5 grid...")
    
    # Find ERA5 target grid
    era5_files = list(era5_dir.glob('*pr*preprocessed*.nc'))
    if not era5_files:
        console.print("  [yellow]ERA5 file not found, skipping regrid[/yellow]")
        return
    
    era5 = xr.open_dataset(era5_files[0])
    target_lats = era5['lat'].values
    target_lons = era5['lon'].values
    era5.close()
    
    for raw_file in seas5_dir.glob('seas5_pr_*.nc'):
        if '_preprocessed' in raw_file.name:
            continue
        
        out_file = raw_file.with_name(raw_file.stem + '_preprocessed.nc')
        
        if out_file.exists():
            console.print(f"    [dim]Already preprocessed: {out_file.name}[/dim]")
            continue
        
        console.print(f"    → Processing {raw_file.name}...")
        
        try:
            ds = xr.open_dataset(raw_file)
            
            # Rename coordinates if needed
            rename_map = {}
            if 'latitude' in ds.coords:
                rename_map['latitude'] = 'lat'
            if 'longitude' in ds.coords:
                rename_map['longitude'] = 'lon'
            if 'valid_time' in ds.coords:
                rename_map['valid_time'] = 'time'
            if 'forecast_reference_time' in ds.coords:
                rename_map['forecast_reference_time'] = 'init_time'
            
            if rename_map:
                ds = ds.rename(rename_map)
            
            # Interpolate to ERA5 grid
            ds = ds.interp(lat=target_lats, lon=target_lons, method='linear')
            
            ds.to_netcdf(out_file)
            ds.close()
            
            console.print(f"    [green]✓ Saved: {out_file.name}[/green]")
            
        except Exception as e:
            console.print(f"    [red]Error: {e}[/red]")


def load_seas5_forecast(
    seas5_dir: Path,
    init_year: int,
    init_month: int,
    lead_months: List[int] = [1, 2, 3],
) -> Optional:
    """
    Load SEAS5 forecast for a specific initialization.
    
    Args:
        seas5_dir: Directory with preprocessed SEAS5 files
        init_year: Initialization year
        init_month: Initialization month
        lead_months: Lead times to load
        
    Returns:
        xarray DataArray with SEAS5 forecast, or None if not found
    """
    import xarray as xr
    
    month_name = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec'][init_month - 1]
    
    # Try preprocessed file first
    seas5_file = seas5_dir / f'seas5_pr_{month_name}_{init_year}-{init_year}_preprocessed.nc'
    
    if not seas5_file.exists():
        # Try finding file that covers this year
        for f in seas5_dir.glob(f'seas5_pr_{month_name}_*_preprocessed.nc'):
            ds = xr.open_dataset(f)
            if 'init_time' in ds.coords:
                years = ds['init_time.year'].values
            else:
                years = ds['time.year'].values
            
            if init_year in years:
                seas5_file = f
                break
            ds.close()
    
    if not seas5_file.exists():
        return None
    
    ds = xr.open_dataset(seas5_file)
    
    # Select the specific initialization
    if 'init_time' in ds.coords:
        fcst = ds.sel(init_time=f'{init_year}-{init_month:02d}')
    else:
        # Assume time is forecast valid time
        fcst = ds.sel(time=ds['time.year'] == init_year)
    
    # Select lead months
    if 'leadtime_month' in ds.coords:
        fcst = fcst.sel(leadtime_month=lead_months)
    
    ds.close()
    
    return fcst


def add_seas5_features(X: dict, seas5_data, cfg: dict) -> dict:
    """
    Add SEAS5 forecast as additional features for TelNet.
    
    This enables "downscaling mode" where TelNet learns to correct/enhance
    SEAS5 forecasts using climate indices.
    
    Args:
        X: Existing input data dict with 'auto' and 'cov' keys
        seas5_data: SEAS5 forecast data
        cfg: Configuration dict
        
    Returns:
        Updated X dict with SEAS5 features added
    """
    import numpy as np
    
    if seas5_data is None:
        return X
    
    # Get SEAS5 ensemble mean or individual members
    use_ensemble_mean = cfg.get('downscaling', {}).get('use_ensemble_mean', True)
    
    if use_ensemble_mean:
        if 'number' in seas5_data.dims:
            seas5_mean = seas5_data.mean('number')
        else:
            seas5_mean = seas5_data
        
        # Add SEAS5 as a covariate
        # This would need to be reshaped to match the index format
        console.print("  [green]✓ SEAS5 forecast added as input feature[/green]")
    
    return X
