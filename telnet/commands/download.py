"""Download command - fetch ERA5, SEAS5, and climate indices data."""

import os
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

console = Console()


def run_download(
    era5: bool,
    seas5: bool,
    indices: bool,
    dates: str,
    config: Path,
):
    """
    Execute the download command.
    
    Args:
        era5: Download ERA5 reanalysis data
        seas5: Download SEAS5 forecast data
        indices: Download climate indices
        dates: Date range in YYYY-YYYY format
        config: Path to configuration file
    """
    from telnet.config import load_config
    
    cfg = load_config(config)
    
    # Parse date range
    start_year, end_year = dates.split("-")
    start_year, end_year = int(start_year), int(end_year)
    
    # Get data directory
    data_dir = Path(os.environ.get('TELNET_DATADIR', './data'))
    data_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel.fit(
        f"[bold blue]TelNet Data Download[/bold blue]\n"
        f"Date range: {start_year} - {end_year}\n"
        f"Region: {cfg['region']['name']}\n"
        f"Data dir: {data_dir}"
    ))
    
    if not any([era5, seas5, indices]):
        console.print("[yellow]No data sources selected. Use --era5, --seas5, or --indices[/yellow]")
        return
    
    if indices:
        download_ersst(data_dir, start_year, end_year)
    
    if era5:
        download_era5(cfg, data_dir, start_year, end_year)
    
    if seas5:
        download_seas5(cfg, data_dir, start_year, end_year)
    
    console.print("\n[bold green]✓ Download complete![/bold green]")


def download_ersst(data_dir: Path, start_year: int, end_year: int):
    """Download ERSSTv5 SST data from NOAA PSL."""
    import wget
    import xarray as xr
    
    console.print("\n[bold]Downloading ERSSTv5 SST data...[/bold]")
    
    suffix = 'present' if end_year > 2024 else str(end_year)
    output_file = data_dir / f"ersstv5_{start_year}-{suffix}.nc"
    
    if output_file.exists():
        console.print(f"  [dim]Already exists: {output_file.name}[/dim]")
        return
    
    download_url = 'https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc'
    tmp_file = data_dir / 'ersstv5_tmp.nc'
    
    console.print(f"  → Downloading from NOAA PSL...")
    wget.download(download_url, str(tmp_file), bar=None)
    
    console.print(f"\n  → Subsetting to {start_year}-{end_year}...")
    ds = xr.open_dataset(tmp_file)
    ds = ds.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))
    if 'time_bnds' in ds:
        ds = ds.drop('time_bnds')
    ds.to_netcdf(output_file)
    ds.close()
    
    tmp_file.unlink()
    console.print(f"  [green]✓ Saved: {output_file.name}[/green]")


def download_era5(cfg: dict, data_dir: Path, start_year: int, end_year: int):
    """Download ERA5 reanalysis data from CDS."""
    import cdsapi
    import xarray as xr
    import numpy as np
    
    console.print("\n[bold]Downloading ERA5 reanalysis data...[/bold]")
    
    # Check for CDS credentials
    cdsapirc = Path.home() / '.cdsapirc'
    if not cdsapirc.exists():
        console.print("[red]ERROR: ~/.cdsapirc not found![/red]")
        console.print("Set up CDS API credentials from: https://cds.climate.copernicus.eu/how-to-api")
        return
    
    era5_dir = data_dir / 'era5'
    era5_dir.mkdir(parents=True, exist_ok=True)
    
    # Target grid (2 degree for indices computation)
    lat2interp = np.arange(-88., 90., 2.0)[::-1]
    lon2interp = np.arange(0., 360., 2.0)
    
    # Region bounds
    region = cfg['region']
    
    # Define variables to download
    single_level_vars = [
        ('10m_u_component_of_wind', 'u10'),
        ('10m_v_component_of_wind', 'v10'),
        ('total_precipitation', 'tp'),
        ('land_sea_mask', 'lsm'),
    ]
    
    pressure_level_vars = [
        ('geopotential', 'z', [500, 700, 1000]),
    ]
    
    years = [str(y) for y in range(start_year, end_year + 1)]
    months = [f'{m:02d}' for m in range(1, 13)]
    
    c = cdsapi.Client()
    
    # Download single-level variables
    for cds_name, var_name in single_level_vars:
        output_file = era5_dir / f'era5_{var_name}_{start_year}-{end_year}_raw.nc'
        
        if output_file.exists():
            console.print(f"  [dim]Already exists: {output_file.name}[/dim]")
            continue
        
        console.print(f"  → Downloading {var_name}...")
        
        try:
            c.retrieve(
                'reanalysis-era5-single-levels-monthly-means',
                {
                    'product_type': 'monthly_averaged_reanalysis',
                    'variable': cds_name,
                    'year': years,
                    'month': months,
                    'time': '00:00',
                    'format': 'netcdf',
                },
                str(output_file)
            )
            console.print(f"  [green]✓ Downloaded: {output_file.name}[/green]")
        except Exception as e:
            console.print(f"  [red]✗ Failed: {e}[/red]")
    
    # Download pressure-level variables
    for cds_name, var_name, levels in pressure_level_vars:
        for level in levels:
            output_file = era5_dir / f'era5_{var_name}_{level}_{start_year}-{end_year}_raw.nc'
            
            if output_file.exists():
                console.print(f"  [dim]Already exists: {output_file.name}[/dim]")
                continue
            
            console.print(f"  → Downloading {var_name} at {level}hPa...")
            
            try:
                c.retrieve(
                    'reanalysis-era5-pressure-levels-monthly-means',
                    {
                        'product_type': 'monthly_averaged_reanalysis',
                        'variable': cds_name,
                        'pressure_level': str(level),
                        'year': years,
                        'month': months,
                        'time': '00:00',
                        'format': 'netcdf',
                    },
                    str(output_file)
                )
                console.print(f"  [green]✓ Downloaded: {output_file.name}[/green]")
            except Exception as e:
                console.print(f"  [red]✗ Failed: {e}[/red]")
    
    console.print("\n  → Preprocessing ERA5 data...")
    preprocess_era5(cfg, era5_dir, start_year, end_year)


def preprocess_era5(cfg: dict, era5_dir: Path, start_year: int, end_year: int):
    """Preprocess ERA5 data: regrid and rename variables."""
    import xarray as xr
    import numpy as np
    
    # Target 2-degree grid
    lat2interp = np.arange(-88., 90., 2.0)[::-1]
    lon2interp = np.arange(0., 360., 2.0)
    
    suffix = f'{start_year}-{end_year}'
    
    for var in ['u10', 'v10', 'tp', 'lsm']:
        raw_file = era5_dir / f'era5_{var}_{suffix}_raw.nc'
        out_file = era5_dir / f'era5_{var}_{suffix}_preprocessed.nc'
        
        if not raw_file.exists():
            continue
        if out_file.exists():
            console.print(f"    [dim]Already preprocessed: {out_file.name}[/dim]")
            continue
        
        console.print(f"    → Preprocessing {var}...")
        ds = xr.open_dataset(raw_file)
        
        # Rename coordinates
        rename_map = {}
        if 'latitude' in ds.coords:
            rename_map['latitude'] = 'lat'
        if 'longitude' in ds.coords:
            rename_map['longitude'] = 'lon'
        if 'valid_time' in ds.coords:
            rename_map['valid_time'] = 'time'
        
        if rename_map:
            ds = ds.rename(rename_map)
        
        # Interpolate to target grid
        ds = ds.interp(lat=lat2interp, lon=lon2interp, method='linear')
        
        ds.to_netcdf(out_file)
        ds.close()
        console.print(f"    [green]✓ Saved: {out_file.name}[/green]")


def download_seas5(cfg: dict, data_dir: Path, start_year: int, end_year: int):
    """Download SEAS5 forecast data from CDS."""
    from telnet.seas5 import download_seas5 as seas5_download
    
    console.print("\n[bold]Downloading SEAS5 forecast data...[/bold]")
    
    region = cfg.get('region', None)
    init_months = [2, 5, 8, 11]  # Feb, May, Aug, Nov (FMA, MJJ, SON, DJF seasons)
    
    seas5_download(
        data_dir=data_dir,
        start_year=start_year,
        end_year=end_year,
        init_months=init_months,
        lead_months=[1, 2, 3, 4, 5, 6],
        region=region,
    )
