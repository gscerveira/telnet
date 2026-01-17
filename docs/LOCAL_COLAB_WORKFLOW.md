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
- Virtual ERA5 store creation (5-10 min)
- ERSSTv5 download (varies)
- Shapefile download (fast)
- Climate indices computation (5-10 min)
- Feature pre-selection (10-30 min depending on samples)

### Phase 2: Sync to Google Drive

```bash
python sync_to_drive.py --source ./telnet_data
```

Follow the instructions to upload to Google Drive.

### Phase 3: Colab GPU Workflow

1. Open `colab_gpu_only_workflow.ipynb` in Google Colab
2. Verify data was synced correctly (Cell 1)
3. Run remaining GPU-intensive cells:
   - Model selection (2-3 hours on T4 GPU)
   - Model testing (30-60 min)
   - Forecast generation (10-20 min)

### Time Savings

| Task | Old (All Colab) | New (Split) |
|------|-----------------|-------------|
| Virtual stores | 5-10 min Colab | 5-10 min Local |
| Climate indices | 5-10 min Colab | 5-10 min Local |
| Feature selection | 10-30 min Colab | 10-30 min Local |
| Model selection | 2-3 hr Colab GPU | 2-3 hr Colab GPU |
| **Total Colab time** | **3-4+ hours** | **2-3 hours** |

This saves ~1 hour of Colab GPU time per run.
