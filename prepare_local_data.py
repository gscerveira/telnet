#!/usr/bin/env python3
"""
Prepare local data for TELNET workflow.

This script runs CPU-intensive data preparation tasks locally, so only GPU work
happens in Colab. It orchestrates existing scripts in the telnet repo:

1. Build virtual ERA5 stores (Icechunk + VirtualiZarr) - streams from S3, no download
2. Download ERSSTv5 sea surface temperature data (for climate indices)
3. Download Maranhao shapefile
4. Compute climate indices (from ERSSTv5 data)
5. Run feature pre-selection (PMI ranking) - uses virtual stores for ERA5

After running this script, upload the output directory to Google Drive
for use in the Colab notebook.
"""

import argparse
import os
import shutil
import subprocess
import sys


def print_banner(step_num: int, total_steps: int, message: str) -> None:
    """Print a step banner for progress tracking."""
    print()
    print("=" * 60)
    print(f"  Step {step_num}/{total_steps}: {message}")
    print("=" * 60)
    print()


def run_command(cmd: list, cwd: str, description: str) -> bool:
    """Run a subprocess command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    print()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            text=True,
        )
        print()
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"[ERROR] {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare local data for TELNET workflow (CPU-intensive tasks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python prepare_local_data.py --output-dir ./telnet_data
  python prepare_local_data.py --output-dir ./telnet_data --skip-virtual
  python prepare_local_data.py --output-dir ./telnet_data --n-samples 50

After completion, upload the output directory to Google Drive for Colab.
        """
    )
    parser.add_argument(
        "--output-dir",
        default="./telnet_data",
        help="Output directory for data (default: ./telnet_data)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for feature pre-selection (default: 100)"
    )
    parser.add_argument(
        "--init-date",
        default="194001",
        help="Initial date YYYYMM (default: 194001)"
    )
    parser.add_argument(
        "--final-date",
        default="202512",
        help="Final date YYYYMM (default: 202512)"
    )
    parser.add_argument(
        "--skip-virtual",
        action="store_true",
        help="Skip building virtual stores"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading ERSSTv5 and shapefile"
    )
    parser.add_argument(
        "--skip-indices",
        action="store_true",
        help="Skip computing climate indices"
    )
    parser.add_argument(
        "--skip-feature-selection",
        action="store_true",
        help="Skip feature pre-selection"
    )

    args = parser.parse_args()

    # Get absolute path for output directory
    output_dir = os.path.abspath(args.output_dir)

    # Get the telnet repo directory (where this script is located)
    telnet_dir = os.path.dirname(os.path.abspath(__file__))

    # Set environment variable
    os.environ["TELNET_DATADIR"] = output_dir
    print(f"TELNET_DATADIR set to: {output_dir}")

    # Create necessary subdirectories
    subdirs = ["shapefiles", "models", "virtual_stores"]
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        print(f"Created directory: {subdir_path}")

    # Calculate total steps based on what is enabled
    steps_enabled = []
    if not args.skip_virtual:
        steps_enabled.append("virtual")
    if not args.skip_download:
        steps_enabled.append("download_ersst")
        steps_enabled.append("download_shapefile")
    if not args.skip_indices:
        steps_enabled.append("indices")
    if not args.skip_feature_selection:
        steps_enabled.append("feature_selection")

    total_steps = len(steps_enabled)
    current_step = 0
    results = {}

    # Step 1: Build virtual ERA5 stores
    if not args.skip_virtual:
        current_step += 1
        print_banner(current_step, total_steps, "Building virtual ERA5 stores")
        cmd = [
            sys.executable,
            os.path.join(telnet_dir, "build_virtual_era5.py"),
            "--initdate", args.init_date,
            "--finaldate", args.final_date,
        ]
        results["virtual"] = run_command(
            cmd, telnet_dir, "Virtual ERA5 store creation"
        )
    else:
        print("\n[SKIP] Building virtual stores (--skip-virtual)")

    # Step 2: Download ERSSTv5 data
    if not args.skip_download:
        current_step += 1
        print_banner(current_step, total_steps, "Downloading ERSSTv5 data")
        # Import and call the download_ersstv5 function directly
        try:
            # Add telnet_dir to path to import the module
            if telnet_dir not in sys.path:
                sys.path.insert(0, telnet_dir)
            from download_preprocess_data import download_ersstv5

            init_date = f"{args.init_date[:4]}-{args.init_date[4:6]}-01"
            final_date = f"{args.final_date[:4]}-{args.final_date[4:6]}-01"
            download_ersstv5(init_date, final_date)
            print()
            print("[OK] ERSSTv5 download completed successfully")
            results["download_ersst"] = True
        except Exception as e:
            print(f"[ERROR] ERSSTv5 download failed: {e}")
            results["download_ersst"] = False

        # Step 3: Download Maranhao shapefile
        current_step += 1
        print_banner(current_step, total_steps, "Downloading Maranhao shapefile")
        cmd = [
            sys.executable,
            os.path.join(telnet_dir, "download_maranhao_shapefile.py"),
        ]
        results["download_shapefile"] = run_command(
            cmd, telnet_dir, "Maranhao shapefile download"
        )
    else:
        print("\n[SKIP] Downloading ERSSTv5 and shapefile (--skip-download)")

    # Step 4: Compute climate indices
    if not args.skip_indices:
        current_step += 1
        print_banner(current_step, total_steps, "Computing climate indices")
        cmd = [
            sys.executable,
            os.path.join(telnet_dir, "compute_climate_indices.py"),
            "--finaldate", args.final_date,
        ]
        results["indices"] = run_command(
            cmd, telnet_dir, "Climate indices computation"
        )
    else:
        print("\n[SKIP] Computing climate indices (--skip-indices)")

    # Step 5: Run feature pre-selection
    if not args.skip_feature_selection:
        current_step += 1
        print_banner(current_step, total_steps, "Running feature pre-selection")
        cmd = [
            sys.executable,
            os.path.join(telnet_dir, "feature_pre_selection.py"),
            "--number", str(args.n_samples),
        ]
        results["feature_selection"] = run_command(
            cmd, telnet_dir, "Feature pre-selection"
        )

        # Copy feature selection outputs from repo to output directory
        # feature_pre_selection.py saves to {repo}/data/, but we need files in {output_dir}/data/
        repo_data_dir = os.path.join(telnet_dir, 'data')
        output_data_dir = os.path.join(output_dir, 'data')
        os.makedirs(os.path.join(output_data_dir, 'models'), exist_ok=True)

        # Copy final_feats.txt
        src_feats = os.path.join(repo_data_dir, 'models', 'final_feats.txt')
        dst_feats = os.path.join(output_data_dir, 'models', 'final_feats.txt')
        if os.path.exists(src_feats):
            shutil.copy2(src_feats, dst_feats)
            print(f"  Copied {src_feats} -> {dst_feats}")

        # Copy seeds_pmi.txt
        src_seeds = os.path.join(repo_data_dir, 'seeds_pmi.txt')
        dst_seeds = os.path.join(output_data_dir, 'seeds_pmi.txt')
        if os.path.exists(src_seeds):
            shutil.copy2(src_seeds, dst_seeds)
            print(f"  Copied {src_seeds} -> {dst_seeds}")
    else:
        print("\n[SKIP] Feature pre-selection (--skip-feature-selection)")

    # Print summary
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print()
    print(f"Output directory: {output_dir}")
    print()

    if results:
        print("Results:")
        for step, success in results.items():
            status = "[OK]" if success else "[FAILED]"
            print(f"  {status} {step}")
        print()

    all_success = all(results.values()) if results else True

    if all_success:
        print("All steps completed successfully!")
        print()
        print("Next steps:")
        print("  1. Upload the output directory to Google Drive:")
        print(f"     {output_dir}")
        print("  2. Open the Colab notebook and mount Google Drive")
        print("  3. Set TELNET_DATADIR to point to the uploaded directory")
        print("  4. Run the GPU-intensive tasks in Colab")
    else:
        print("Some steps failed. Please check the output above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
