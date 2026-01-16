"""
Main orchestrator for running TelNet seasonal forecasts for Maranhão.

This script coordinates the full workflow:
1. Download data (ERA5 from AWS, ERSSTv5 from NOAA)
2. Compute climate indices
3. Run feature pre-selection
4. Run model selection
5. Generate forecasts for specified initialization dates
6. Extract Maranhão region
7. Verify against observations

Usage:
    python run_maranhao_forecast.py --init-dates 202501 202504 202507 202510
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime


def run_command(cmd, description):
    """Run a shell command with logging."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"COMPLETED: {description}")
    return result


def setup_environment():
    """Verify environment variables are set."""
    required_vars = ['TELNET_DATADIR']

    for var in required_vars:
        if not os.getenv(var):
            print(f"ERROR: Environment variable {var} not set")
            print(f"Please set it: export {var}=/path/to/data")
            sys.exit(1)

    # Create data directories
    datadir = os.getenv('TELNET_DATADIR')
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(os.path.join(datadir, 'era5'), exist_ok=True)
    os.makedirs(os.path.join(datadir, 'shapefiles'), exist_ok=True)
    os.makedirs(os.path.join(datadir, 'models'), exist_ok=True)

    print(f"Data directory: {datadir}")


def download_data(init_date='194001', final_date='202512'):
    """Download all required data."""

    # Download ERA5 from AWS
    run_command(
        f"python download_era5_aws.py -idate {init_date} -fdate {final_date}",
        "Download ERA5 from AWS"
    )

    # Download ERSSTv5 (uses existing script logic)
    run_command(
        f"python -c \"from download_preprocess_data import download_ersstv5; download_ersstv5('{init_date[:4]}-{init_date[4:]}-01', '{final_date[:4]}-{final_date[4:]}-01')\"",
        "Download ERSSTv5 from NOAA"
    )

    # Download Maranhão shapefile
    run_command(
        "python download_maranhao_shapefile.py",
        "Download Maranhão shapefile"
    )


def compute_indices(final_date='202512'):
    """Compute climate indices."""
    run_command(
        f"python compute_climate_indices.py -fdate {final_date}",
        "Compute climate indices"
    )


def run_feature_selection(n_samples=100):
    """Run feature pre-selection."""
    run_command(
        f"python feature_pre_selection.py -n {n_samples}",
        "Feature pre-selection"
    )


def run_model_selection(n_samples=100, n_gpus=1):
    """Run model selection grid search."""
    run_command(
        f"./model_selection.sh {n_samples} {n_gpus}",
        "Model selection"
    )


def run_model_testing(n_samples=100, config=1):
    """Run model testing with selected configuration."""
    run_command(
        f"python model_testing.py -n {n_samples} -c {config}",
        "Model testing"
    )


def generate_forecast(init_date, config=1):
    """Generate forecast for a single initialization date."""
    run_command(
        f"./generate_forecast.sh {init_date} {config}",
        f"Generate forecast for {init_date}"
    )


def extract_maranhao(init_date):
    """Extract Maranhão region from forecast."""
    datadir = os.getenv('TELNET_DATADIR')
    results_dir = os.path.join(datadir, 'results', init_date)
    shapefile_dir = os.path.join(datadir, 'shapefiles')

    # Find forecast file
    forecast_files = [f for f in os.listdir(results_dir) if f.endswith('.nc')]

    for f in forecast_files:
        input_path = os.path.join(results_dir, f)
        output_path = os.path.join(results_dir, f'maranhao_{f}')

        run_command(
            f"python extract_maranhao.py {input_path} {output_path} --shapefile-dir {shapefile_dir}",
            f"Extract Maranhão from {f}"
        )


def verify_forecast(init_date):
    """Verify forecast against observations."""
    datadir = os.getenv('TELNET_DATADIR')
    results_dir = os.path.join(datadir, 'results', init_date)
    shapefile_dir = os.path.join(datadir, 'shapefiles')

    # Forecast and observation paths
    forecast_path = os.path.join(results_dir, 'maranhao_ensemble.nc')
    obs_path = os.path.join(datadir, 'era5', 'era5_pr_2025-2025_preprocessed.nc')
    output_dir = os.path.join(results_dir, 'verification')

    run_command(
        f"python verify_forecasts.py {forecast_path} {obs_path} -o {output_dir} --shapefile-dir {shapefile_dir}",
        f"Verify forecast {init_date}"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Run TelNet seasonal forecasts for Maranhão'
    )
    parser.add_argument(
        '--init-dates', nargs='+', default=['202501', '202504', '202507', '202510'],
        help='Initialization dates (YYYYMM format)'
    )
    parser.add_argument(
        '--skip-download', action='store_true',
        help='Skip data download step'
    )
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Skip feature selection and model training'
    )
    parser.add_argument(
        '--n-samples', type=int, default=100,
        help='Number of samples for feature selection and model selection'
    )
    parser.add_argument(
        '--config', type=int, default=1,
        help='Model configuration to use'
    )
    parser.add_argument(
        '--virtual',
        action='store_true',
        help='Use virtual Icechunk stores instead of downloading data'
    )

    args = parser.parse_args()

    print("="*60)
    print("TelNet Maranhão Seasonal Forecast Workflow")
    print("="*60)
    print(f"Initialization dates: {args.init_dates}")
    print(f"Samples: {args.n_samples}")
    print(f"Config: {args.config}")
    print(f"Virtual mode: {args.virtual}")

    # Setup
    setup_environment()

    # Data download
    if not args.skip_download:
        datadir = os.getenv('TELNET_DATADIR')
        # Determine final date from init_dates
        final_date = max(args.init_dates)

        if args.virtual:
            print("Using virtual ERA5 stores (streaming from S3)...")
            # Build virtual stores if they don't exist
            virtual_dir = os.path.join(datadir, 'virtual_stores')
            if not os.path.exists(virtual_dir):
                subprocess.run([
                    'python', 'build_virtual_era5.py',
                    '-idate', '194001', '-fdate', final_date
                ], check=True)
        else:
            download_data()
        compute_indices()

    # Model training
    if not args.skip_training:
        run_feature_selection(args.n_samples)
        run_model_selection(args.n_samples)
        run_model_testing(args.n_samples, args.config)

    # Generate forecasts for each initialization date
    for init_date in args.init_dates:
        print(f"\n{'#'*60}")
        print(f"Processing initialization: {init_date}")
        print(f"{'#'*60}")

        generate_forecast(init_date, args.config)
        extract_maranhao(init_date)
        verify_forecast(init_date)

    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {os.path.join(os.getenv('TELNET_DATADIR'), 'results')}")


if __name__ == "__main__":
    main()
