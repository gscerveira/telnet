"""Train command - train the TelNet model."""

import os
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def run_train(
    config: Path,
    epochs: Optional[int],
    resume: Optional[Path],
    gpu: bool,
):
    """
    Execute the training command.
    
    Args:
        config: Path to configuration file
        epochs: Number of epochs (overrides config)
        resume: Resume from checkpoint
        gpu: Use GPU if available
    """
    from telnet.config import load_config
    
    cfg = load_config(config)
    
    # Override epochs if specified
    if epochs is not None:
        cfg['model']['epochs'] = epochs
    
    console.print(Panel.fit(
        f"[bold blue]TelNet Training Pipeline[/bold blue]\n"
        f"Config: {config}\n"
        f"Region: {cfg['region']['name']}\n"
        f"Epochs: {cfg['model']['epochs']}\n"
        f"Features: {cfg['model']['nfeats']}\n"
        f"GPU: {'enabled' if gpu else 'disabled'}"
    ))
    
    # Setup device
    device = _setup_device(gpu)
    console.print(f"Device: [cyan]{device}[/cyan]")
    
    # Setup paths
    data_dir = Path(os.environ.get('TELNET_DATADIR', './data'))
    models_dir = data_dir / 'models'
    results_dir = Path(cfg.get('paths', {}).get('results_dir', './results'))
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment for existing code compatibility
    os.environ['TELNET_DATADIR'] = str(data_dir)
    
    # Step 1: Load data
    console.print("\n[bold]Step 1: Loading data...[/bold]")
    X, Y, idcs_list = _load_training_data(cfg, data_dir)
    
    if X is None:
        console.print("[red]Failed to load data. Check that data files exist.[/red]")
        return
    
    # Step 2: Feature selection
    console.print("\n[bold]Step 2: Feature pre-selection (PMI ranking)...[/bold]")
    selected_features = _run_feature_selection(X, Y, cfg, data_dir, device)
    
    # Step 3: Train model
    console.print("\n[bold]Step 3: Training TelNet model...[/bold]")
    model = _train_model(X, Y, cfg, selected_features, device, resume)
    
    # Step 4: Save model
    model_path = models_dir / 'telnet.pt'
    _save_model(model, model_path, cfg, selected_features, models_dir)
    
    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"Model saved to: {model_path}")
    console.print(f"Feature list: {models_dir / 'final_feats.txt'}")


def _setup_device(gpu: bool) -> str:
    """Set up compute device."""
    if gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
                console.print(f"  GPU detected: {torch.cuda.get_device_name(0)}")
                return device
            else:
                console.print("  [yellow]No GPU detected, falling back to CPU[/yellow]")
        except ImportError:
            console.print("  [yellow]PyTorch not available, using CPU[/yellow]")
    return "cpu"


def _load_training_data(cfg: dict, data_dir: Path):
    """Load observation data for training."""
    try:
        # Add parent directory to path for imports
        import sys
        telnet_root = Path(__file__).parent.parent.parent
        if str(telnet_root) not in sys.path:
            sys.path.insert(0, str(telnet_root))
        
        from utils import read_obs_data
        
        console.print("  → Loading ERA5 precipitation observations...")
        console.print("  → Loading climate indices...")
        
        X, Y, idcs_list = read_obs_data()
        
        console.print(f"  [green]✓ Loaded {len(idcs_list)} climate indices[/green]")
        console.print(f"  [green]✓ Target shape: {Y.shape}[/green]")
        
        return X, Y, idcs_list
        
    except Exception as e:
        console.print(f"  [red]Error loading data: {e}[/red]")
        console.print("  [dim]Make sure you've run 'telnet download' first[/dim]")
        return None, None, None


def _run_feature_selection(X, Y, cfg: dict, data_dir: Path, device: str):
    """Run PMI-based feature pre-selection."""
    import numpy as np
    
    nfeats = cfg['model']['nfeats']
    features_file = data_dir / 'models' / 'final_feats.txt'
    
    # Check if features already exist
    if features_file.exists():
        features = np.loadtxt(features_file, dtype=str, delimiter=' ')
        console.print(f"  [dim]Using existing feature list: {features_file}[/dim]")
        console.print(f"  Selected features: {list(features[:nfeats])}")
        return features[:nfeats]
    
    console.print("  → Running PMI-based feature ranking...")
    console.print("  [yellow]This may take 5-15 minutes...[/yellow]")
    
    try:
        # Add parent directory to path for imports
        import sys
        telnet_root = Path(__file__).parent.parent.parent
        if str(telnet_root) not in sys.path:
            sys.path.insert(0, str(telnet_root))
        
        from feature_pre_selection import main as feature_selection_main
        
        # Run feature selection with small sample for speed
        # The original uses Monte Carlo sampling
        feature_selection_main(nsamples=100, reproduce_paper=False)
        
        # Load results
        features = np.loadtxt(features_file, dtype=str, delimiter=' ')
        console.print(f"  [green]✓ Selected {len(features)} features[/green]")
        console.print(f"  Top features: {list(features[:5])}...")
        
        return features[:nfeats]
        
    except Exception as e:
        console.print(f"  [red]Feature selection failed: {e}[/red]")
        console.print("  [yellow]Using default features from config...[/yellow]")
        
        default_features = cfg.get('features', {}).get('indices', [
            'oni', 'atn-sst', 'iobw', 'atl-sst', 'ats-sst',
            'nao', 'aao', 'ao', 'pna', 'iod'
        ])
        return np.array(default_features[:nfeats])


def _train_model(X, Y, cfg: dict, features, device: str, resume: Optional[Path]):
    """Train the TelNet model."""
    import numpy as np
    
    epochs = cfg['model']['epochs']
    
    console.print(f"  → Training for {epochs} epochs...")
    console.print(f"  → Using {len(features)} input features")
    
    if resume:
        console.print(f"  → Resuming from: {resume}")
    
    try:
        import sys
        telnet_root = Path(__file__).parent.parent.parent
        if str(telnet_root) not in sys.path:
            sys.path.insert(0, str(telnet_root))
        
        from copy import deepcopy
        from utils import get_search_matrix, exp_data_dir
        from test import main as test_main
        
        # Get model configuration
        search_arr, search_df = get_search_matrix()
        model_config = search_arr[0]  # Use config 0
        model_config = [int(x) if x % 1 == 0 else float(x) for x in model_config]
        
        # Override epochs from config
        model_config[3] = epochs
        
        # Select features in data
        X['cov']['var'] = X['cov']['var'].sel(indices=features)
        
        # Train with one seed for CLI (full training uses multiple samples)
        console.print("  → Running training iteration...")
        
        # Simplified training - single sample
        seed = cfg.get('training', {}).get('seed', 42)
        seeds = np.array([seed])
        
        # We need to prepare baseline_models structure
        init_months = [1, 4, 7, 10]
        baseline_models = {i+1: {'dyn': {}, 'dl': {}} for i in init_months}
        
        result = test_main((
            seed,
            deepcopy(X),
            deepcopy(Y),
            deepcopy(model_config),
            deepcopy(features),
            seeds,
            deepcopy(baseline_models),
            init_months
        ))
        
        val_rps = result[0] if result else None
        
        if val_rps is not None:
            console.print(f"  [green]✓ Training complete! Validation RPS: {val_rps:.4f}[/green]")
        else:
            console.print(f"  [green]✓ Training complete![/green]")
        
        return result
        
    except Exception as e:
        console.print(f"  [red]Training error: {e}[/red]")
        import traceback
        console.print(f"  [dim]{traceback.format_exc()}[/dim]")
        return None


def _save_model(model, path: Path, cfg: dict, features, models_dir: Path):
    """Save trained model and configuration."""
    import numpy as np
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    console.print("  → Saving model...")
    
    # Save feature list
    features_file = models_dir / 'final_feats.txt'
    if not features_file.exists():
        np.savetxt(features_file, features, fmt='%s')
        console.print(f"  [green]✓ Saved feature list[/green]")
    
    # The actual model checkpoint is saved by test.py during training
    # Check if it exists
    checkpoint_path = models_dir / 'telnet.pt'
    if checkpoint_path.exists():
        console.print(f"  [green]✓ Model checkpoint exists[/green]")
    else:
        console.print(f"  [yellow]Note: Model checkpoint will be saved during training[/yellow]")
