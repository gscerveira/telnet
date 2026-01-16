# Colab Workflow Notebook Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Jupyter notebook that runs the entire Maranhão forecast workflow directly in Google Colab's interface (no SSH required).

**Architecture:** Single notebook with sequential cells for each workflow step - clone repo, install deps, download data, train model, generate forecasts, verify results. Each major step is a separate cell so users can monitor progress and resume if disconnected.

**Tech Stack:** Jupyter notebook, Python, Google Colab, Google Drive (for persistence)

---

## Task 1: Create the Colab Workflow Notebook

**Files:**
- Create: `colab_maranhao_workflow.ipynb`

**Step 1: Create the notebook with all cells**

Create a Jupyter notebook with the following structure:

```python
# Cell 1 - Setup: Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

# Create persistent data directory
import os
TELNET_DATADIR = '/content/drive/MyDrive/telnet_data'
os.makedirs(TELNET_DATADIR, exist_ok=True)
os.makedirs(f'{TELNET_DATADIR}/era5', exist_ok=True)
os.makedirs(f'{TELNET_DATADIR}/shapefiles', exist_ok=True)
os.makedirs(f'{TELNET_DATADIR}/models', exist_ok=True)
os.environ['TELNET_DATADIR'] = TELNET_DATADIR
print(f"Data directory: {TELNET_DATADIR}")
```

```python
# Cell 2 - Clone repository and install dependencies
%cd /content
!rm -rf telnet  # Clean previous install if exists
!git clone https://github.com/gscerveira/telnet.git
%cd telnet

# Install dependencies
!pip install -q -r docker/requirements.txt
!pip install -q s3fs zarr geopandas rioxarray

print("Dependencies installed!")
```

```python
# Cell 3 - Verify GPU
!nvidia-smi
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

```python
# Cell 4 - Download ERA5 data from AWS
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'

# This downloads precipitation, winds, geopotential, and land-sea mask
# Takes ~30-60 minutes depending on date range
!python download_era5_aws.py -idate 194001 -fdate 202512
```

```python
# Cell 5 - Download ERSSTv5 (sea surface temperature)
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'

from download_preprocess_data import download_ersstv5
download_ersstv5('1940-01-01', '2025-12-01')
print("ERSSTv5 downloaded!")
```

```python
# Cell 6 - Download Maranhão shapefile
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'

!python download_maranhao_shapefile.py
print("Shapefile downloaded!")
```

```python
# Cell 7 - Compute climate indices
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'

!python compute_climate_indices.py -fdate 202512
```

```python
# Cell 8 - Feature pre-selection
# Adjust n_samples based on available time (100 is faster, 1000 is more robust)
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'

!python feature_pre_selection.py -n 100
```

```python
# Cell 9 - Model selection (GPU intensive)
# This is the longest step - can take several hours
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'

!chmod +x model_selection.sh
!./model_selection.sh 100 1
```

```python
# Cell 10 - Model testing
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'

!python model_testing.py -n 100 -c 1
```

```python
# Cell 11 - Generate forecasts for 2025
# Run for each initialization month
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'

!chmod +x generate_forecast.sh

# January 2025 initialization
!./generate_forecast.sh 202501 1

# April 2025 initialization
!./generate_forecast.sh 202504 1

# July 2025 initialization
!./generate_forecast.sh 202507 1

# October 2025 initialization
!./generate_forecast.sh 202510 1
```

```python
# Cell 12 - Extract Maranhão region from forecasts
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'
DATADIR = os.environ['TELNET_DATADIR']

import glob

# Find all forecast files and extract Maranhão region
for init_date in ['202501', '202504', '202507', '202510']:
    results_dir = f'{DATADIR}/results/{init_date}'
    if os.path.exists(results_dir):
        for f in glob.glob(f'{results_dir}/*.nc'):
            if not f.startswith('maranhao_'):
                output = f'{results_dir}/maranhao_{os.path.basename(f)}'
                !python extract_maranhao.py "{f}" "{output}" --shapefile-dir {DATADIR}/shapefiles
                print(f"Extracted: {output}")
```

```python
# Cell 13 - Verify forecasts against observations
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'
DATADIR = os.environ['TELNET_DATADIR']

for init_date in ['202501', '202504', '202507', '202510']:
    results_dir = f'{DATADIR}/results/{init_date}'
    forecast_file = f'{results_dir}/maranhao_ensemble.nc'
    obs_file = f'{DATADIR}/era5/era5_pr_2025-2025_preprocessed.nc'
    output_dir = f'{results_dir}/verification'

    if os.path.exists(forecast_file):
        !python verify_forecasts.py "{forecast_file}" "{obs_file}" -o "{output_dir}" --shapefile-dir {DATADIR}/shapefiles
        print(f"Verification complete for {init_date}")
```

```python
# Cell 14 - View results
%cd /content/telnet
import os
os.environ['TELNET_DATADIR'] = '/content/drive/MyDrive/telnet_data'
DATADIR = os.environ['TELNET_DATADIR']

import pandas as pd
from IPython.display import display, Image

# Show skill scores
for init_date in ['202501', '202504', '202507', '202510']:
    skills_file = f'{DATADIR}/results/{init_date}/verification/skill_scores.csv'
    if os.path.exists(skills_file):
        print(f"\n=== {init_date} Skill Scores ===")
        display(pd.read_csv(skills_file))

    # Show reliability diagram if exists
    diagram = f'{DATADIR}/results/{init_date}/verification/reliability_diagrams.png'
    if os.path.exists(diagram):
        display(Image(diagram))
```

**Step 2: Verify notebook is valid JSON**

Run: `python -c "import json; json.load(open('colab_maranhao_workflow.ipynb'))"`
Expected: No output (valid JSON)

**Step 3: Commit**

```bash
git add colab_maranhao_workflow.ipynb
git commit -m "feat: add Colab workflow notebook for Maranhão forecasts

Runs entire workflow in Colab interface without SSH:
- Mounts Google Drive for persistence
- Downloads ERA5 from AWS, ERSSTv5, shapefiles
- Runs feature selection and model training
- Generates forecasts for all 2025 init dates
- Extracts Maranhão region and verifies

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Update documentation

**Files:**
- Modify: `docs/COLAB_SSH_SETUP.md`

**Step 1: Add note about alternative notebook**

Add at the top of the file, after the title:

```markdown
> **Note:** If you prefer to work directly in Colab's interface without SSH, use `colab_maranhao_workflow.ipynb` instead. It runs the entire workflow in notebook cells with Google Drive persistence.
```

**Step 2: Commit**

```bash
git add docs/COLAB_SSH_SETUP.md
git commit -m "docs: add reference to non-SSH Colab workflow notebook

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

After completing these tasks, you will have:

| File | Purpose |
|------|---------|
| `colab_maranhao_workflow.ipynb` | Complete workflow notebook for Colab (no SSH) |

**Usage:**
1. Upload `colab_maranhao_workflow.ipynb` to Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Run cells sequentially
4. Data persists in Google Drive between sessions
