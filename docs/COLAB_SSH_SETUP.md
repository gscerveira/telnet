# Google Colab SSH Setup

> **Warning:** Google Colab has recently restricted SSH tunneling on their platform. You may experience disconnections or account restrictions. **Recommended alternative:** Use `colab_maranhao_workflow.ipynb` instead, which runs the entire workflow directly in Colab's notebook interface with Google Drive persistence.

This guide explains how to run TelNet on Google Colab while working from your local terminal.

## Prerequisites

- Google account
- Google Colab access (free tier works, Pro recommended for longer sessions)
- SSH client on your local machine

## Step 1: Open the Setup Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Upload `colab_ssh_setup.ipynb` from this repository
3. Change runtime to GPU: Runtime → Change runtime type → T4 GPU

## Step 2: Run the Setup Cells

Run each cell in order:

1. **Cell 1**: Installs cloudflared
2. **Cell 2**: Sets root password and starts SSH (note the password!)
3. **Cell 3**: Starts the tunnel (copy the URL that appears)

## Step 3: Connect from Your Terminal

```bash
# Install cloudflared locally if not already installed
# macOS: brew install cloudflared
# Ubuntu: wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared-linux-amd64.deb

# Connect via SSH
ssh -o ProxyCommand="cloudflared access ssh --hostname %h" root@<tunnel-url>
```

Enter the password shown in Cell 2 when prompted.

## Step 4: Setup TelNet Environment

Once connected:

```bash
# Clone the repository
git clone https://github.com/<your-repo>/telnet.git
cd telnet

# Install dependencies
pip install -r docker/requirements.txt
pip install s3fs zarr geopandas rioxarray

# Set environment variables
export TELNET_DATADIR=/content/data
mkdir -p $TELNET_DATADIR

# Mount Google Drive for persistence (optional)
# Run this in a Colab cell first: from google.colab import drive; drive.mount('/content/drive')
# Then link: ln -s /content/drive/MyDrive/telnet_data $TELNET_DATADIR
```

## Step 5: Run the Workflow

```bash
# Full workflow
python run_maranhao_forecast.py --init-dates 202501 202504 202507 202510

# Or step by step:
python download_era5_aws.py -idate 194001 -fdate 202512
python download_maranhao_shapefile.py
python compute_climate_indices.py -fdate 202512
# ... etc
```

## Session Limits

| Tier | Max Session | Idle Timeout | GPU |
|------|-------------|--------------|-----|
| Free | ~12 hours | ~90 min | T4 |
| Pro | ~24 hours | ~90 min | T4/A100 |
| Pro+ | ~24 hours | ~90 min | A100 priority |

## Tips

- **Save progress frequently**: Push to git or sync to Drive
- **Use screen/tmux**: Keep processes running if SSH disconnects
- **Check GPU**: Run `nvidia-smi` to verify GPU is available
- **Resume training**: Scripts are designed to skip completed steps

## Troubleshooting

**Tunnel disconnects**: Re-run Cell 3 and reconnect

**Session dies**: Your files are lost. Always save to Drive or git.

**Out of memory**: Reduce batch size in model_selection.py
