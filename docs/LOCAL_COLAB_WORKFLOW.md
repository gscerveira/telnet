## Two-Phase Workflow: Local + Colab

To save Colab GPU minutes, the workflow is split into two phases:

### Phase 1: Local Preparation (CPU-only, no GPU needed)

Run on your local machine:

```bash
# Set up environment
export TELNET_DATADIR=./telnet_data

# Run all CPU-intensive preparation
python prepare_local_data.py --output-dir ./telnet_data --n-samples 100
```

This runs:
- **Virtual ERA5 store creation** (~30-40 min first time only) - scans S3 metadata, no data download
- ERSSTv5 download (for climate indices)
- Shapefile download (fast)
- Climate indices computation
- Feature pre-selection

**Important:** Virtual stores only need to be built once. Use `--skip-virtual` on subsequent runs.
After building, ERA5 data streams directly from S3 during training - no local storage needed.

### Phase 2: Sync to Google Drive

```bash
python sync_to_drive.py --source ./telnet_data
```

Follow the instructions to upload to Google Drive.

### Phase 3: Colab GPU Workflow

1. Open `colab_gpu_only_workflow.ipynb` in Google Colab
2. Verify data was synced correctly (Cell 1)
3. Run remaining GPU-intensive cells:
   - Model selection
   - Model testing
   - Forecast generation

### Benefits of Virtual Stores

| Before | After |
|--------|-------|
| Download ~50GB ERA5 | Build virtual stores once (~30-40 min) |
| Read from local files | Stream from S3 on-demand |
| Requires disk space | Only ~50MB metadata |
