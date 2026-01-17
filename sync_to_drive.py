#!/usr/bin/env python3
"""
Sync helper script for uploading TELNET data to Google Drive.

After running prepare_local_data.py locally, use this script to get instructions
for uploading the prepared data to Google Drive for use in the Colab notebook.

Provides three sync options:
1. Manual Upload - Using Google Drive web interface
2. Using rclone - Command-line tool for cloud storage
3. Using Google Drive Desktop - File manager copy
"""

import argparse
import os
import sys


def get_size_str(size_bytes: int) -> str:
    """Convert bytes to human-readable size string."""
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


def get_dir_size(path: str) -> int:
    """Calculate total size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, FileNotFoundError):
                pass
    return total_size


def get_file_size(path: str) -> int:
    """Get size of a file in bytes."""
    try:
        return os.path.getsize(path)
    except (OSError, FileNotFoundError):
        return 0


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Helper script for syncing prepared TELNET data to Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python sync_to_drive.py
  python sync_to_drive.py --source ./telnet_data
  python sync_to_drive.py --source /path/to/custom/data

This script provides instructions for uploading data prepared by
prepare_local_data.py to Google Drive for use in the Colab notebook.
        """
    )
    parser.add_argument(
        "--source",
        default="./telnet_data",
        help="Source directory with prepared data (default: ./telnet_data)"
    )

    args = parser.parse_args()

    # Get absolute path for source directory
    source_dir = os.path.abspath(args.source)

    # Check if source directory exists
    if not os.path.isdir(source_dir):
        print(f"[ERROR] Source directory not found: {source_dir}")
        print()
        print("Please run prepare_local_data.py first to generate the data:")
        print("  python prepare_local_data.py --output-dir ./telnet_data")
        sys.exit(1)

    print_section("TELNET Data Sync Helper")
    print(f"Source directory: {source_dir}")

    # Define items that need to be synced
    sync_items = [
        ("virtual_stores/", "Virtual ERA5 references (Icechunk stores)"),
        ("era5/", "ERA5 data files"),
        ("shapefiles/", "Region boundary shapefiles"),
        ("data/models/", "Feature selection results"),
        ("data/seeds_pmi.txt", "Random seeds for reproducibility"),
        ("seasonal_climate_indices_1941-2024.txt", "Computed climate indices"),
    ]

    # Calculate sizes and check existence
    print_section("Files/Directories to Sync")

    total_size = 0
    items_found = []
    items_missing = []

    for item_path, description in sync_items:
        full_path = os.path.join(source_dir, item_path)

        if os.path.exists(full_path):
            if os.path.isdir(full_path):
                size = get_dir_size(full_path)
            else:
                size = get_file_size(full_path)

            total_size += size
            items_found.append((item_path, description, size))
            print(f"  [OK] {item_path:<45} {get_size_str(size):>12}")
            print(f"        {description}")
        else:
            items_missing.append((item_path, description))
            print(f"  [--] {item_path:<45} {'Not found':>12}")
            print(f"        {description}")

    print()
    print(f"Total size to sync: {get_size_str(total_size)}")

    if items_missing:
        print()
        print(f"Note: {len(items_missing)} item(s) not found. They may not have been")
        print("generated yet or may be optional for your workflow.")

    if not items_found:
        print()
        print("[WARNING] No data files found to sync!")
        print("Please run prepare_local_data.py first.")
        sys.exit(1)

    # Option 1: Manual Upload
    print_section("OPTION 1: Manual Upload (Google Drive Web Interface)")
    print("1. Open Google Drive in your browser: https://drive.google.com")
    print()
    print("2. Create a folder named 'telnet_data' (or your preferred name)")
    print()
    print("3. Upload the following items from:")
    print(f"   {source_dir}")
    print()
    for item_path, description, size in items_found:
        print(f"   - {item_path}")
    print()
    print("4. Right-click the folder and select 'Get link' to verify it's accessible")

    # Option 2: Using rclone
    print_section("OPTION 2: Using rclone (Command Line)")
    print("If you haven't set up rclone yet:")
    print("  1. Install rclone: https://rclone.org/install/")
    print("  2. Configure Google Drive: rclone config")
    print("     - Choose 'n' for new remote")
    print("     - Name it 'gdrive'")
    print("     - Choose 'Google Drive' as storage type")
    print("     - Follow OAuth prompts")
    print()
    print("Then run these commands to sync:")
    print()
    print("# Create destination folder")
    print("rclone mkdir gdrive:telnet_data")
    print()
    print("# Sync each item")
    for item_path, description, size in items_found:
        # Remove trailing slash for rclone commands
        clean_path = item_path.rstrip("/")
        src_path = os.path.join(source_dir, clean_path)
        if item_path.endswith("/"):
            print(f"rclone sync \"{src_path}\" \"gdrive:telnet_data/{clean_path}\" --progress")
        else:
            print(f"rclone copy \"{src_path}\" \"gdrive:telnet_data/\" --progress")
    print()
    print("# Or sync everything at once:")
    print(f"rclone sync \"{source_dir}\" \"gdrive:telnet_data\" --progress")

    # Option 3: Using Google Drive Desktop
    print_section("OPTION 3: Using Google Drive Desktop (File Manager)")
    print("If you have Google Drive Desktop installed:")
    print("  - Windows: Google Drive appears as a drive letter (e.g., G:)")
    print("  - macOS: Google Drive appears in Finder under Locations")
    print("  - Linux: Use google-drive-ocamlfuse or similar")
    print()
    print("Copy commands (adjust the destination path for your system):")
    print()
    print("# Linux/macOS example (adjust ~/Google\\ Drive path as needed):")
    drive_dest = "~/Google\\ Drive/telnet_data"
    print(f"mkdir -p {drive_dest}")
    for item_path, description, size in items_found:
        clean_path = item_path.rstrip("/")
        src_path = os.path.join(source_dir, clean_path)
        if item_path.endswith("/"):
            print(f"cp -r \"{src_path}\" \"{drive_dest}/{clean_path}\"")
        else:
            # Get parent directory for files
            parent = os.path.dirname(clean_path)
            if parent:
                print(f"mkdir -p \"{drive_dest}/{parent}\" && cp \"{src_path}\" \"{drive_dest}/{clean_path}\"")
            else:
                print(f"cp \"{src_path}\" \"{drive_dest}/{clean_path}\"")
    print()
    print("# Windows example (adjust G:\\My Drive path as needed):")
    print("# Use File Explorer to copy the folders to your Google Drive")

    # Final message
    print_section("Next Steps")
    print("After syncing your data to Google Drive:")
    print()
    print("1. Open the Colab notebook:")
    print("   colab_gpu_only_workflow.ipynb")
    print()
    print("2. Mount Google Drive in Colab:")
    print("   from google.colab import drive")
    print("   drive.mount('/content/drive')")
    print()
    print("3. Set the data directory path:")
    print("   TELNET_DATADIR = '/content/drive/MyDrive/telnet_data'")
    print()
    print("4. Run the GPU-intensive tasks in Colab!")
    print()


if __name__ == "__main__":
    main()
