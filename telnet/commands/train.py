"""Train command - train the TelNet model."""

from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel

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
        "[bold blue]TelNet Training[/bold blue]\n"
        f"Config: {config}\n"
        f"Epochs: {cfg['model']['epochs']}\n"
        f"GPU: {'enabled' if gpu else 'disabled'}"
    ))
    
    # Check GPU availability
    device = _setup_device(gpu)
    console.print(f"Device: {device}")
    
    # Load data
    console.print("\n[bold]Loading data...[/bold]")
    X, Y = _load_training_data(cfg)
    
    # Feature selection
    console.print("\n[bold]Running feature selection...[/bold]")
    selected_features = _run_feature_selection(X, Y, cfg)
    
    # Train model
    console.print("\n[bold]Training model...[/bold]")
    model = _train_model(X, Y, selected_features, cfg, device, resume)
    
    # Save model
    output_path = Path(cfg['paths']['models_dir']) / 'telnet.pt'
    _save_model(model, output_path, cfg)
    
    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"Model saved to: {output_path}")


def _setup_device(gpu: bool) -> str:
    """Set up compute device."""
    if gpu:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass
    return "cpu"


def _load_training_data(cfg: dict):
    """Load and prepare training data."""
    # TODO: Implement data loading
    console.print("  → Loading ERA5 precipitation...")
    console.print("  → Loading climate indices...")
    console.print("  → Preparing training samples...")
    
    # Placeholder
    return None, None


def _run_feature_selection(X, Y, cfg: dict):
    """Run PMI-based feature pre-selection."""
    # TODO: Implement feature selection
    indices = cfg.get('features', {}).get('indices', [])
    console.print(f"  → Selected {len(indices)} features")
    return indices


def _train_model(X, Y, features, cfg: dict, device: str, resume: Optional[Path]):
    """Train the TelNet model."""
    # TODO: Implement training loop
    epochs = cfg['model']['epochs']
    
    for epoch in range(1, min(3, epochs) + 1):  # Just show a few for demo
        console.print(f"  Epoch {epoch}/{epochs} - Loss: 0.XXX")
    
    if epochs > 3:
        console.print(f"  ...")
    
    return None  # Placeholder


def _save_model(model, path: Path, cfg: dict):
    """Save trained model and configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # TODO: Implement model saving
    console.print(f"  → Saving model weights...")
    console.print(f"  → Saving feature list...")
