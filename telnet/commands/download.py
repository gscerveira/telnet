"""Download command - fetch ERA5, SEAS5, and climate indices data."""

from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

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
    
    console.print(f"\n[bold blue]TelNet Data Download[/bold blue]")
    console.print(f"Date range: {start_year} - {end_year}")
    console.print(f"Region: {cfg['region']['name']}")
    console.print()
    
    if not any([era5, seas5, indices]):
        console.print("[yellow]No data sources selected. Use --era5, --seas5, or --indices[/yellow]")
        return
    
    if era5:
        _download_era5(cfg, start_year, end_year)
    
    if seas5:
        _download_seas5(cfg, start_year, end_year)
    
    if indices:
        _download_indices(cfg, start_year, end_year)
    
    console.print("\n[bold green]✓ Download complete![/bold green]")


def _download_era5(cfg: dict, start_year: int, end_year: int):
    """Download ERA5 reanalysis data from CDS."""
    console.print("[bold]Downloading ERA5 reanalysis...[/bold]")
    
    # TODO: Implement ERA5 download using cdsapi
    # For now, print instructions
    console.print("  → Precipitation (tp)")
    console.print("  → 10m wind components (u10, v10)")
    console.print("  → Geopotential heights (z500, z700, z1000)")
    console.print()
    console.print("[yellow]Note: ERA5 download requires CDS API credentials.[/yellow]")
    console.print("Set up ~/.cdsapirc with your credentials from:")
    console.print("https://cds.climate.copernicus.eu/how-to-api")
    

def _download_seas5(cfg: dict, start_year: int, end_year: int):
    """Download SEAS5 forecast data from CDS."""
    console.print("[bold]Downloading SEAS5 forecasts...[/bold]")
    
    # TODO: Implement SEAS5 download
    console.print("  → Monthly precipitation forecasts")
    console.print("  → Lead times: 1-6 months")
    console.print()
    console.print("[yellow]SEAS5 download will be implemented in Phase 2.[/yellow]")


def _download_indices(cfg: dict, start_year: int, end_year: int):
    """Download climate indices from NOAA/ERSST."""
    console.print("[bold]Downloading climate indices...[/bold]")
    
    indices = cfg.get('features', {}).get('indices', [])
    for idx in indices:
        console.print(f"  → {idx}")
    
    # TODO: Implement index download from NOAA
    console.print()
    console.print("[yellow]Climate indices will be computed from SST data.[/yellow]")
