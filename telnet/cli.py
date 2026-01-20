"""
TelNet CLI - Command-line interface for seasonal precipitation forecasting.

Usage:
    telnet download --era5 --dates 2020-2025
    telnet train --config config.yaml --epochs 200
    telnet forecast --date 202501 --output forecasts/
    telnet evaluate --compare seas5,ccsm4 --metric rps
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel

# Create main app
app = typer.Typer(
    name="telnet",
    help="TelNet - Teleconnection-based Neural Network for Seasonal Precipitation Forecasting",
    add_completion=False,
)

console = Console()


def version_callback(value: bool):
    if value:
        console.print("[bold blue]TelNet[/bold blue] version [green]0.1.0[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
):
    """
    TelNet - Seasonal precipitation forecasting using teleconnection indices.
    
    A machine learning model that predicts seasonal precipitation by learning
    relationships between climate indices (ENSO, NAO, etc.) and regional rainfall.
    """
    pass


@app.command()
def download(
    era5: bool = typer.Option(False, "--era5", help="Download ERA5 reanalysis data"),
    seas5: bool = typer.Option(False, "--seas5", help="Download SEAS5 forecast data"),
    indices: bool = typer.Option(False, "--indices", help="Download climate indices"),
    dates: str = typer.Option(
        "2020-2024",
        "--dates",
        "-d",
        help="Date range in YYYY-YYYY format",
    ),
    config: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """
    Download required data for training and forecasting.
    
    Examples:
        telnet download --era5 --dates 2020-2024
        telnet download --seas5 --indices
    """
    from telnet.commands.download import run_download
    run_download(era5=era5, seas5=seas5, indices=indices, dates=dates, config=config)


@app.command()
def train(
    config: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Number of training epochs (overrides config)",
    ),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume training from checkpoint",
    ),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use GPU if available"),
):
    """
    Train the TelNet model.
    
    Examples:
        telnet train --config config/default.yaml
        telnet train --epochs 100 --no-gpu
    """
    from telnet.commands.train import run_train
    run_train(config=config, epochs=epochs, resume=resume, gpu=gpu)


@app.command()
def forecast(
    date: str = typer.Argument(..., help="Forecast initialization date (YYYYMM format)"),
    config: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    output: Path = typer.Option(
        Path("results/forecasts"),
        "--output",
        "-o",
        help="Output directory for forecasts",
    ),
    model: Optional[Path] = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to trained model (default: use config)",
    ),
):
    """
    Generate seasonal precipitation forecast.
    
    Examples:
        telnet forecast 202501
        telnet forecast 202504 --output my_forecasts/
    """
    from telnet.commands.forecast import run_forecast
    run_forecast(date=date, config=config, output=output, model=model)


@app.command()
def evaluate(
    compare: str = typer.Option(
        "seas5",
        "--compare",
        help="Comma-separated list of baseline models to compare",
    ),
    metric: str = typer.Option(
        "rps",
        "--metric",
        help="Evaluation metric (rps, rmse, correlation)",
    ),
    config: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    years: str = typer.Option(
        "2000-2020",
        "--years",
        help="Test year range",
    ),
):
    """
    Evaluate TelNet against baseline models.
    
    Examples:
        telnet evaluate --compare seas5,ccsm4 --metric rps
        telnet evaluate --years 2010-2020
    """
    from telnet.commands.evaluate import run_evaluate
    run_evaluate(compare=compare, metric=metric, config=config, years=years)


if __name__ == "__main__":
    app()
