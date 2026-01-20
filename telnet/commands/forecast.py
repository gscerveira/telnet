"""Forecast command - generate seasonal precipitation forecasts."""

from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

console = Console()


def run_forecast(
    date: str,
    config: Path,
    output: Path,
    model: Optional[Path],
):
    """
    Execute the forecast command.
    
    Args:
        date: Forecast initialization date (YYYYMM format)
        config: Path to configuration file
        output: Output directory for forecasts
        model: Path to trained model
    """
    from telnet.config import load_config
    
    cfg = load_config(config)
    
    # Parse date
    year = int(date[:4])
    month = int(date[4:6])
    
    console.print(f"\n[bold blue]TelNet Seasonal Forecast[/bold blue]")
    console.print(f"Initialization: {year}-{month:02d}")
    console.print(f"Region: {cfg['region']['name']}")
    console.print()
    
    # Load model
    model_path = model or Path(cfg['paths']['models_dir']) / 'telnet.pt'
    console.print(f"Loading model: {model_path}")
    telnet_model = _load_model(model_path, cfg)
    
    # Load input data
    console.print("\n[bold]Loading input data...[/bold]")
    X = _load_forecast_inputs(cfg, year, month)
    
    # Generate forecast
    console.print("\n[bold]Generating forecasts...[/bold]")
    forecasts = _run_inference(telnet_model, X, cfg)
    
    # Save outputs
    output.mkdir(parents=True, exist_ok=True)
    _save_forecasts(forecasts, output, year, month, cfg)
    
    # Display summary
    _display_summary(forecasts, year, month)
    
    console.print(f"\n[bold green]✓ Forecast complete![/bold green]")
    console.print(f"Output saved to: {output}")


def _load_model(path: Path, cfg: dict):
    """Load trained TelNet model."""
    # TODO: Implement model loading
    console.print("  → Loading model weights...")
    return None  # Placeholder


def _load_forecast_inputs(cfg: dict, year: int, month: int):
    """Load climate indices and prepare inputs for forecasting."""
    console.print("  → Loading climate indices...")
    console.print("  → Loading historical precipitation...")
    
    # Check if SEAS5 downscaling is enabled
    if cfg.get('downscaling', {}).get('enabled', False):
        console.print("  → Loading SEAS5 forecast (downscaling mode)...")
    
    return None  # Placeholder


def _run_inference(model, X, cfg: dict):
    """Run model inference to generate forecasts."""
    lead_times = cfg.get('model', {}).get('lead', 6)
    nmembs = cfg.get('model', {}).get('nmembs', 10)
    
    console.print(f"  → Generating {nmembs} ensemble members...")
    console.print(f"  → Lead times: 1-{lead_times} months")
    
    return {}  # Placeholder


def _save_forecasts(forecasts: dict, output: Path, year: int, month: int, cfg: dict):
    """Save forecast outputs as NetCDF and PNG."""
    console.print("  → Saving NetCDF data...")
    console.print("  → Generating forecast maps...")
    

def _display_summary(forecasts: dict, year: int, month: int):
    """Display forecast summary table."""
    table = Table(title="Forecast Summary")
    table.add_column("Lead", style="cyan")
    table.add_column("Target Month", style="magenta")
    table.add_column("Below Normal", style="blue")
    table.add_column("Normal", style="green")
    table.add_column("Above Normal", style="red")
    
    # Placeholder data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for lead in range(1, 5):
        target_month = (month + lead - 1) % 12
        table.add_row(
            f"{lead}",
            months[target_month],
            "33%",
            "34%",
            "33%",
        )
    
    console.print(table)
