"""Evaluate command - compare TelNet against baseline models."""

from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def run_evaluate(
    compare: str,
    metric: str,
    config: Path,
    years: str,
):
    """
    Execute the evaluation command.
    
    Args:
        compare: Comma-separated list of baseline models
        metric: Evaluation metric (rps, rmse, correlation)
        config: Path to configuration file
        years: Test year range
    """
    from telnet.config import load_config
    
    cfg = load_config(config)
    
    # Parse inputs
    baselines = [m.strip() for m in compare.split(",")]
    start_year, end_year = years.split("-")
    start_year, end_year = int(start_year), int(end_year)
    
    console.print(f"\n[bold blue]TelNet Model Evaluation[/bold blue]")
    console.print(f"Metric: {metric.upper()}")
    console.print(f"Test period: {start_year}-{end_year}")
    console.print(f"Baselines: {', '.join(baselines)}")
    console.print()
    
    # Load observations
    console.print("[bold]Loading observations...[/bold]")
    obs = _load_observations(cfg, start_year, end_year)
    
    # Load TelNet predictions
    console.print("\n[bold]Loading TelNet predictions...[/bold]")
    telnet_preds = _load_telnet_predictions(cfg, start_year, end_year)
    
    # Load baseline predictions
    console.print("\n[bold]Loading baseline predictions...[/bold]")
    baseline_preds = {}
    for baseline in baselines:
        baseline_preds[baseline] = _load_baseline(cfg, baseline, start_year, end_year)
    
    # Compute metrics
    console.print(f"\n[bold]Computing {metric.upper()}...[/bold]")
    results = _compute_metrics(obs, telnet_preds, baseline_preds, metric, cfg)
    
    # Display results
    _display_results(results, metric)


def _load_observations(cfg: dict, start_year: int, end_year: int):
    """Load observed precipitation for verification."""
    console.print("  → Loading ERA5 precipitation...")
    return None  # Placeholder


def _load_telnet_predictions(cfg: dict, start_year: int, end_year: int):
    """Load TelNet hindcast predictions."""
    console.print("  → Loading TelNet hindcasts...")
    return None  # Placeholder


def _load_baseline(cfg: dict, name: str, start_year: int, end_year: int):
    """Load baseline model predictions."""
    console.print(f"  → Loading {name}...")
    return None  # Placeholder


def _compute_metrics(obs, telnet_preds, baseline_preds, metric: str, cfg: dict) -> dict:
    """Compute evaluation metrics for all models."""
    results = {}
    
    # Placeholder values - will be replaced with actual computation
    results['TelNet'] = 0.45
    results['Climatology'] = 0.67
    
    for name in baseline_preds:
        # Placeholder
        if name.lower() == 'seas5':
            results[name] = 0.37
        else:
            results[name] = 0.55
    
    return results


def _display_results(results: dict, metric: str):
    """Display evaluation results in a formatted table."""
    table = Table(title=f"Model Comparison ({metric.upper()})")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Model", style="magenta")
    table.add_column(metric.upper(), style="green")
    table.add_column("Skill vs Clim", style="yellow")
    table.add_column("", style="blue")
    
    # Sort by metric (lower is better for RPS)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    clim_score = results.get('Climatology', 0.67)
    
    for rank, (model, score) in enumerate(sorted_results, 1):
        skill = (clim_score - score) / clim_score * 100 if clim_score > 0 else 0
        marker = "⭐ YOUR MODEL" if model == 'TelNet' else ""
        bar = "█" * int(max(0, (1 - score) * 15))
        
        table.add_row(
            str(rank),
            model,
            f"{score:.4f}",
            f"{skill:+.1f}%",
            f"{bar} {marker}",
        )
    
    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Lower RPS is better. Skill > 0% means better than climatology.[/dim]")
