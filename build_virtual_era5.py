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
    """Get list of precipitation files for date range."""
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yyyymm = f"{year}{month:02d}"
            # Large-scale precipitation
            lsp = fs.glob(f"{BUCKET}/e5.oper.fc.sfc.accumu/{yyyymm}/e5.oper.fc.sfc.accumu.128_142_lsp.ll025sc.*.nc")
            # Convective precipitation
            cp = fs.glob(f"{BUCKET}/e5.oper.fc.sfc.accumu/{yyyymm}/e5.oper.fc.sfc.accumu.128_143_cp.ll025sc.*.nc")
            files.extend([f"s3://{f}" for f in lsp + cp])
    return sorted(files)


def get_wind_files(fs, start_year, end_year, component='10u'):
    """Get list of wind files for date range."""
    var_code = '128_165_10u' if component == '10u' else '128_166_10v'
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yyyymm = f"{year}{month:02d}"
            matches = fs.glob(f"{BUCKET}/e5.oper.an.sfc/{yyyymm}/e5.oper.an.sfc.{var_code}.ll025sc.*.nc")
            files.extend([f"s3://{f}" for f in matches])
    return sorted(files)


def get_geopotential_files(fs, start_year, end_year):
    """Get list of geopotential files for date range."""
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yyyymm = f"{year}{month:02d}"
            matches = fs.glob(f"{BUCKET}/e5.oper.an.pl/{yyyymm}/e5.oper.an.pl.128_129_z.ll025sc.*.nc")
            files.extend([f"s3://{f}" for f in matches])
    return sorted(files)


def build_virtual_store(files, store_path, variable_name):
    """
    Build a virtual Icechunk store from a list of NetCDF files.
    """
    import time

    print(f"\n{'='*60}")
    print(f"Building virtual store for: {variable_name}")
    print(f"Total files to process: {len(files)}")
    print(f"Output: {store_path}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Create virtual datasets from each file
    virtual_datasets = []
    errors = 0
    for i, url in enumerate(files):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            if i > 0:
                rate = i / elapsed
                remaining = (len(files) - i) / rate
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                print(f"  [{i+1:4d}/{len(files)}] {(i/len(files)*100):5.1f}% | "
                      f"Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | "
                      f"ETA: {eta_min}m {eta_sec}s | Errors: {errors}")
            else:
                print(f"  [{i+1:4d}/{len(files)}] Starting...")
        try:
            vds = open_virtual_dataset(
                url,
                parser=_parser,
                registry=_registry,
                loadable_variables=[]
            )
            virtual_datasets.append(vds)
        except (OSError, ValueError, IOError) as e:
            errors += 1
            if errors <= 5:  # Only show first 5 errors
                print(f"  Warning: Could not virtualize {url}: {e}")
            continue

    if not virtual_datasets:
        print(f"  ERROR: No files virtualized for {variable_name}")
        return None

    elapsed = time.time() - start_time
    print(f"\n  Scan complete: {len(virtual_datasets)} files in {int(elapsed//60)}m {int(elapsed%60)}s")
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

    fs = fsspec.filesystem('s3', anon=True)

    variables = [args.variable] if args.variable != 'all' else ['precipitation', 'u10', 'v10', 'geopotential']

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
