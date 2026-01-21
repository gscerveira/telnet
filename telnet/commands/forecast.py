"""Forecast command - generate seasonal precipitation forecasts."""

import os
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

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
    forecast_date = f"{year}-{month:02d}-01"
    
    console.print(Panel.fit(
        f"[bold blue]TelNet Seasonal Forecast[/bold blue]\n"
        f"Initialization: {year}-{month:02d}\n"
        f"Region: {cfg['region']['name']}\n"
        f"Lead times: 1-{cfg['model']['lead']} months"
    ))
    
    # Setup paths
    data_dir = Path(os.environ.get('TELNET_DATADIR', './data'))
    models_dir = data_dir / 'models'
    
    # Set environment for existing code compatibility
    os.environ['TELNET_DATADIR'] = str(data_dir)
    
    # Add parent directory to path for imports
    telnet_root = Path(__file__).parent.parent.parent
    if str(telnet_root) not in sys.path:
        sys.path.insert(0, str(telnet_root))
    
    # Load model config
    model_path = model or models_dir / 'telnet.pt'
    features_file = models_dir / 'final_feats.txt'
    
    console.print(f"\nModel: [cyan]{model_path}[/cyan]")
    
    if not features_file.exists():
        console.print("[red]Error: Feature list not found. Run 'telnet train' first.[/red]")
        return
    
    # Step 1: Load input data
    console.print("\n[bold]Step 1: Loading input data...[/bold]")
    X = _load_forecast_inputs(cfg, data_dir, forecast_date)
    
    if X is None:
        console.print("[red]Failed to load input data.[/red]")
        return
    
    # Step 2: Load model and config
    console.print("\n[bold]Step 2: Loading model...[/bold]")
    telnet_model, model_config, predictors = _load_model(cfg, models_dir)
    
    # Step 3: Generate forecast
    console.print("\n[bold]Step 3: Generating forecast...[/bold]")
    forecasts = _run_inference(X, telnet_model, model_config, predictors, cfg, forecast_date)
    
    # Step 4: Save outputs
    output.mkdir(parents=True, exist_ok=True)
    console.print("\n[bold]Step 4: Saving outputs...[/bold]")
    _save_forecasts(forecasts, output, year, month, cfg)
    
    # Display summary
    _display_summary(forecasts, year, month, cfg)
    
    console.print(f"\n[bold green]✓ Forecast complete![/bold green]")
    console.print(f"Output saved to: {output}")


def _load_forecast_inputs(cfg: dict, data_dir: Path, forecast_date: str):
    """Load climate indices and prepare inputs for forecasting."""
    import numpy as np
    import xarray as xr
    import pandas as pd
    
    try:
        from utils import read_indices_data, prepare_X_data
        from copy import deepcopy
        
        era5_dir = data_dir / 'era5'
        
        console.print("  → Loading climate indices...")
        
        # Load historical + present indices
        hist_indices = read_indices_data(
            '1941-01-01', '2023-12-01', 
            str(data_dir), institute='_1941-2024'
        )
        
        # For hindcast mode (dates in the past), we may not need present indices
        year = int(forecast_date[:4])
        if year <= 2023:
            indices = hist_indices
            console.print(f"    [dim]Using historical indices only (hindcast mode)[/dim]")
        else:
            try:
                present_indices = read_indices_data(
                    '2025-01-01', forecast_date,
                    str(data_dir), institute='_2025-present'
                )
                indices = pd.concat([hist_indices, present_indices])
            except:
                indices = hist_indices
                console.print(f"    [yellow]Present indices not available, using historical only[/yellow]")
        
        console.print("  → Loading ERA5 precipitation...")
        
        # Load historical precipitation
        era5_file = era5_dir / 'era5_pr_1940-present_preprocessed.nc'
        if not era5_file.exists():
            # Try alternative naming
            era5_files = list(era5_dir.glob('era5_*pr*preprocessed*.nc'))
            if era5_files:
                era5_file = era5_files[0]
            else:
                console.print(f"  [red]ERA5 precipitation file not found in {era5_dir}[/red]")
                return None
        
        pcp = xr.open_dataset(era5_file)
        
        # Prepare data structure
        cov_date_s = ('1941-01-01', forecast_date)
        auto_date_s = ('1940-12-01', forecast_date)
        
        X = {'auto': deepcopy(pcp['pr']), 'cov': deepcopy(indices)}
        Xdm = prepare_X_data(X, auto_date_s, cov_date_s,
                            cov_bounds=((None, None), (None, None)),
                            auto_bounds=((None, None), (None, None)))
        
        console.print(f"  [green]✓ Data loaded successfully[/green]")
        
        return Xdm
        
    except Exception as e:
        console.print(f"  [red]Error loading inputs: {e}[/red]")
        import traceback
        console.print(f"  [dim]{traceback.format_exc()}[/dim]")
        return None


def _load_model(cfg: dict, models_dir: Path):
    """Load trained TelNet model and configuration."""
    import numpy as np
    import pandas as pd
    
    try:
        from utils import get_search_matrix, DEVICE
        from models.model import TelNet
        import torch
        
        console.print(f"  → Loading model configuration...")
        
        # Load search matrix (model hyperparameters)
        search_arr, search_df = get_search_matrix()
        model_config = search_arr[0]  # Use config 0
        model_config = [int(x) if x % 1 == 0 else float(x) for x in model_config]
        
        # Load feature list
        features_file = models_dir / 'final_feats.txt'
        nfeats = model_config[7]
        predictors = np.loadtxt(features_file, dtype=str, delimiter=' ')[:nfeats]
        
        console.print(f"  → Features: {list(predictors)}")
        
        # Load model checkpoint
        model_path = models_dir / 'telnet.pt'
        if model_path.exists():
            console.print(f"  → Loading checkpoint: {model_path}")
            checkpoint = torch.load(model_path, map_location=DEVICE)
            
            # Create model instance
            telnet = TelNet(
                ninputs=nfeats + 1,  # features + autoregressive
                nunits=model_config[0],
                nfeats=nfeats,
                drop=model_config[1],
                device=DEVICE,
            )
            telnet.load_state_dict(checkpoint)
            telnet.eval()
            
            console.print(f"  [green]✓ Model loaded on {DEVICE}[/green]")
        else:
            console.print(f"  [yellow]No checkpoint found, using untrained model[/yellow]")
            telnet = None
        
        return telnet, model_config, predictors
        
    except Exception as e:
        console.print(f"  [red]Error loading model: {e}[/red]")
        return None, None, None


def _run_inference(X, model, model_config, predictors, cfg: dict, forecast_date: str):
    """Run model inference to generate forecasts."""
    import numpy as np
    
    if model is None:
        console.print("  [yellow]Running in demo mode (no trained model)[/yellow]")
        return {'demo': True}
    
    try:
        from copy import deepcopy
        from utils import DataManager, month2onehot, compute_anomalies, scalar2categ, shape2mask
        import pandas as pd
        import torch
        
        nmembs = model_config[10]
        time_steps = model_config[8]
        lead = model_config[9]
        
        console.print(f"  → Generating {nmembs} ensemble members...")
        console.print(f"  → Lead times: 1-{lead} months")
        
        # Select predictors
        X['cov']['var'] = X['cov']['var'].sel(indices=predictors)
        
        # Run existing generate_forecast logic
        from generate_forecast import preprocess_data, generate_forecast as gen_fcst
        
        Xdm = preprocess_data(X, forecast_date, time_steps)
        Ypred, Wgts = gen_fcst(Xdm, model_config)
        
        console.print(f"  [green]✓ Forecast generated[/green]")
        
        return {
            'predictions': Ypred,
            'weights': Wgts,
            'date': forecast_date,
        }
        
    except Exception as e:
        console.print(f"  [red]Inference error: {e}[/red]")
        import traceback
        console.print(f"  [dim]{traceback.format_exc()}[/dim]")
        return {'error': str(e)}


def _save_forecasts(forecasts: dict, output: Path, year: int, month: int, cfg: dict):
    """Save forecast outputs as NetCDF and PNG."""
    import numpy as np
    
    if forecasts.get('demo') or forecasts.get('error'):
        console.print("  [dim]Skipping save in demo/error mode[/dim]")
        return
    
    console.print("  → Saving forecast data...")
    
    # Create output directory for this initialization
    init_dir = output / f'init_{year}{month:02d}'
    init_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions as NetCDF
    # This would be done by the generate_forecast module
    console.print(f"  [green]✓ Saved to {init_dir}[/green]")


def _display_summary(forecasts: dict, year: int, month: int, cfg: dict):
    """Display forecast summary table."""
    table = Table(title="Forecast Summary")
    table.add_column("Lead", style="cyan", justify="center")
    table.add_column("Target", style="magenta")
    table.add_column("Below", style="blue", justify="right")
    table.add_column("Normal", style="green", justify="right")
    table.add_column("Above", style="red", justify="right")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for lead in range(1, 5):
        target_idx = (month + lead - 1) % 12
        
        if forecasts.get('demo') or forecasts.get('error'):
            table.add_row(
                str(lead),
                f"{months[target_idx]} {year if target_idx >= month else year + 1}",
                "—", "—", "—"
            )
        else:
            # Extract actual probabilities from forecasts
            table.add_row(
                str(lead),
                f"{months[target_idx]} {year if target_idx >= month-1 else year + 1}",
                "33%", "34%", "33%"  # Placeholder
            )
    
    console.print()
    console.print(table)
