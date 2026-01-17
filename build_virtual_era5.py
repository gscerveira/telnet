"""
Build a virtual Zarr store for ERA5 data using Icechunk + VirtualiZarr.
References NetCDF files on S3 without downloading them.
"""

import os
import re
import argparse
import fsspec
import xarray as xr
import icechunk
import obstore
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry


BUCKET = "nsf-ncar-era5"
S3_PREFIX = f"s3://{BUCKET}/"

# Set up S3 store and registry for virtualizarr
# nsf-ncar-era5 bucket is in us-west-2
_s3_store = obstore.store.S3Store.from_url(
    S3_PREFIX,
    config={"skip_signature": True, "region": "us-west-2"}
)
_registry = ObjectStoreRegistry({S3_PREFIX: _s3_store})
_parser = HDFParser()


def get_precipitation_files(fs, start_year, end_year):
    """Generate precipitation file URLs directly (no S3 listing needed)."""
    from datetime import date
    from dateutil.relativedelta import relativedelta

    print(f"  Generating precipitation file URLs ({start_year}-{end_year})...")

    # ERA5 precipitation files follow a predictable pattern:
    # Two files per month per variable (LSP and CP):
    # - {YYYY}{MM}0106_{YYYY}{MM}1606.nc (1st-16th of month)
    # - {YYYY}{MM}1606_{YYYY}{MM+1}0106.nc (16th to 1st of next month)

    files = []
    base_path = f"s3://{BUCKET}/e5.oper.fc.sfc.accumu"

    current = date(start_year, 1, 1)
    end = date(end_year, 12, 31)

    while current <= end:
        yyyymm = current.strftime("%Y%m")
        next_month = current + relativedelta(months=1)
        next_yyyymm = next_month.strftime("%Y%m")

        for var_code in ['128_142_lsp', '128_143_cp']:
            # First half of month: 0106 to 1606
            files.append(
                f"{base_path}/{yyyymm}/e5.oper.fc.sfc.accumu.{var_code}.ll025sc.{yyyymm}0106_{yyyymm}1606.nc"
            )
            # Second half: 1606 to next month 0106
            files.append(
                f"{base_path}/{yyyymm}/e5.oper.fc.sfc.accumu.{var_code}.ll025sc.{yyyymm}1606_{next_yyyymm}0106.nc"
            )

        current = next_month

    print(f"    Generated {len(files)} file URLs (no S3 listing needed)")
    return sorted(files)


def get_wind_files(fs, start_year, end_year, component='10u'):
    """Generate wind file URLs directly (no S3 listing needed)."""
    from datetime import date
    from dateutil.relativedelta import relativedelta
    import calendar

    var_code = '128_165_10u' if component == '10u' else '128_166_10v'
    print(f"  Generating {component} wind file URLs ({start_year}-{end_year})...")

    # Wind files: 1 file per month covering full month
    # Pattern: {YYYY}{MM}0100_{YYYY}{MM}{LAST_DAY}23.nc

    files = []
    base_path = f"s3://{BUCKET}/e5.oper.an.sfc"

    current = date(start_year, 1, 1)
    end = date(end_year, 12, 31)

    while current <= end:
        yyyymm = current.strftime("%Y%m")
        last_day = calendar.monthrange(current.year, current.month)[1]

        files.append(
            f"{base_path}/{yyyymm}/e5.oper.an.sfc.{var_code}.ll025sc.{yyyymm}0100_{yyyymm}{last_day:02d}23.nc"
        )

        current = current + relativedelta(months=1)

    print(f"    Generated {len(files)} file URLs (no S3 listing needed)")
    return sorted(files)


def get_geopotential_files(fs, start_year, end_year):
    """Generate geopotential file URLs directly (no S3 listing needed)."""
    from datetime import date, timedelta

    print(f"  Generating geopotential file URLs ({start_year}-{end_year})...")

    # Geopotential files: 1 file per day
    # Pattern: {YYYY}{MM}{DD}00_{YYYY}{MM}{DD}23.nc

    files = []
    base_path = f"s3://{BUCKET}/e5.oper.an.pl"

    current = date(start_year, 1, 1)
    end = date(end_year, 12, 31)

    while current <= end:
        yyyymm = current.strftime("%Y%m")
        yyyymmdd = current.strftime("%Y%m%d")

        files.append(
            f"{base_path}/{yyyymm}/e5.oper.an.pl.128_129_z.ll025sc.{yyyymmdd}00_{yyyymmdd}23.nc"
        )

        current = current + timedelta(days=1)

    print(f"    Generated {len(files)} file URLs (no S3 listing needed)")
    return sorted(files)


def build_virtual_store(files, store_path, variable_name):
    """
    Build a virtual Icechunk store from a list of NetCDF files.
    Uses parallel processing for faster metadata scanning.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"\n{'='*60}")
    print(f"Building virtual store for: {variable_name}")
    print(f"Total files to process: {len(files)}")
    print(f"Output: {store_path}")
    print(f"{'='*60}\n")

    start_time = time.time()

    def virtualize_file(url):
        """Virtualize a single file, return None on error."""
        try:
            return open_virtual_dataset(
                url,
                parser=_parser,
                registry=_registry,
                loadable_variables=[]
            )
        except (OSError, ValueError, IOError):
            return None

    # Parallel virtualization with progress tracking
    print(f"  Virtualizing {len(files)} files (32 parallel workers)...")
    virtual_datasets = []
    errors = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(virtualize_file, url): url for url in files}

        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result is not None:
                virtual_datasets.append(result)
            else:
                errors += 1

            # Progress update every 100 files
            if completed % 100 == 0 or completed == len(files):
                elapsed = time.time() - start_time
                if completed > 0:
                    rate = completed / elapsed
                    remaining = (len(files) - completed) / rate if rate > 0 else 0
                    eta_min = int(remaining // 60)
                    eta_sec = int(remaining % 60)
                    print(f"    [{completed:4d}/{len(files)}] {(completed/len(files)*100):5.1f}% | "
                          f"Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | "
                          f"ETA: {eta_min}m {eta_sec}s | OK: {len(virtual_datasets)} | Errors: {errors}")

    if not virtual_datasets:
        print(f"  ERROR: No files virtualized for {variable_name}")
        return None

    elapsed = time.time() - start_time
    print(f"\n  Virtualization complete: {len(virtual_datasets)} files in {int(elapsed//60)}m {int(elapsed%60)}s")
    if errors > 0:
        print(f"  Skipped {errors} files due to errors")

    # Combine into single virtual dataset
    print(f"\n  Combining {len(virtual_datasets)} virtual datasets into single store...")
    combine_start = time.time()
    virtual_ds = xr.concat(
        virtual_datasets,
        dim='time',
        coords='minimal',
        compat='override',
        combine_attrs='override'
    )
    combine_time = time.time() - combine_start
    print(f"  Combine complete in {int(combine_time)}s")

    # Create Icechunk repository
    print(f"\n  Creating Icechunk repository...")
    storage = icechunk.local_filesystem_storage(path=store_path)
    config = icechunk.RepositoryConfig.default()

    # Configure virtual chunk container for S3
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            S3_PREFIX,
            icechunk.s3_store(region="us-west-2")
        )
    )

    # Set up anonymous S3 credentials
    credentials = icechunk.containers_credentials({
        S3_PREFIX: icechunk.s3_credentials(anonymous=True)
    })

    repo = icechunk.Repository.create(storage, config, credentials)

    # Write virtual dataset to Icechunk
    print(f"  Writing virtual dataset to Icechunk...")
    write_start = time.time()
    session = repo.writable_session("main")
    virtual_ds.virtualize.to_icechunk(session.store)
    snapshot_id = session.commit(f"Initial {variable_name} virtual store")
    write_time = time.time() - write_start
    print(f"  Write complete in {int(write_time)}s")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SUCCESS: Virtual store created for {variable_name}")
    print(f"  Location: {store_path}")
    print(f"  Snapshot: {snapshot_id}")
    print(f"  Total time: {int(total_time//60)}m {int(total_time%60)}s")
    print(f"{'='*60}\n")

    return store_path


def main():
    parser = argparse.ArgumentParser(
        description='Build virtual ERA5 Icechunk stores (no data download)'
    )
    parser.add_argument(
        '-idate', '--initdate',
        help='Initial date YYYYMM (e.g., 194001)',
        required=True
    )
    parser.add_argument(
        '-fdate', '--finaldate',
        help='Final date YYYYMM (e.g., 202512)',
        required=True
    )
    parser.add_argument(
        '--variable',
        choices=['precipitation', 'u10', 'v10', 'geopotential', 'all'],
        default='all',
        help='Which variable to build (default: all)'
    )

    args = parser.parse_args()

    # Validate YYYYMM date format
    yyyymm_pattern = re.compile(r'^\d{4}(0[1-9]|1[0-2])$')
    if not yyyymm_pattern.match(args.initdate):
        raise ValueError(f"Invalid initdate '{args.initdate}': must be YYYYMM format with valid month (01-12)")
    if not yyyymm_pattern.match(args.finaldate):
        raise ValueError(f"Invalid finaldate '{args.finaldate}': must be YYYYMM format with valid month (01-12)")

    start_year = int(args.initdate[:4])
    end_year = int(args.finaldate[:4])

    root_datadir = os.getenv('TELNET_DATADIR', '/content/drive/MyDrive/telnet_data')
    virtual_dir = os.path.join(root_datadir, 'virtual_stores')
    os.makedirs(virtual_dir, exist_ok=True)

    print("Connecting to S3 (anonymous access)...")
    fs = fsspec.filesystem('s3', anon=True)
    print("Connected to S3.\n")

    variables = [args.variable] if args.variable != 'all' else ['precipitation', 'u10', 'v10', 'geopotential']
    print(f"Variables to process: {variables}")
    print(f"Date range: {args.initdate} to {args.finaldate}")
    print()

    for var in variables:
        store_path = os.path.join(virtual_dir, f'era5_{var}')

        if os.path.exists(store_path):
            print(f"Virtual store already exists: {store_path}")
            continue

        files = []
        if var == 'precipitation':
            files = get_precipitation_files(fs, start_year, end_year)
        elif var == 'u10':
            files = get_wind_files(fs, start_year, end_year, '10u')
        elif var == 'v10':
            files = get_wind_files(fs, start_year, end_year, '10v')
        elif var == 'geopotential':
            files = get_geopotential_files(fs, start_year, end_year)

        result = build_virtual_store(files, store_path, var)
        if result is None:
            print(f"Warning: Failed to build virtual store for {var}")

    print("\nVirtual ERA5 stores complete!")
    print(f"Stores saved to: {virtual_dir}")
    print("No data was downloaded - only metadata references were created.")


if __name__ == "__main__":
    main()
