# TelNet

An interpretable machine learning model for seasonal precipitation forecasting.

> Based on the paper ["An interpretable machine learning model for seasonal precipitation forecasting"](https://www.nature.com/articles/s43247-025-02207-2)

## Quick Start

### Using the CLI

```bash
# Install
pip install -e .

# Set data directory
export TELNET_DATADIR=/path/to/data

# Download data
telnet download --era5 --indices --dates 2020-2024

# Train model
telnet train --config config/default.yaml --epochs 200

# Generate forecast
telnet forecast 202501

# Evaluate against baselines
telnet evaluate --compare seas5,ccsm4 --metric rps
```

### Using Docker

```bash
# Training (GPU)
docker compose run train

# Forecasting (CPU)
docker compose run forecast 202501

# Download data
docker compose run download
```

## Commands

| Command | Description |
|---------|-------------|
| `download` | Download ERA5, SEAS5, or climate indices data |
| `train` | Train TelNet model with feature selection |
| `forecast` | Generate seasonal precipitation forecasts |
| `evaluate` | Compare TelNet against baseline models |

## Pipeline

![pipeline](imgs/pipeline.jpg)

1. **Data Acquisition**: Download ERA5 reanalysis and climate indices
2. **Feature Selection**: PMI-based input variable selection
3. **Model Training**: Neural network training with attention
4. **Forecast Generation**: Produce seasonal precipitation forecasts

## Setup

1. **Download baseline models** from [Google Drive](https://drive.google.com/file/d/1EMJr323Oz7j4GIqsV3Bmcv2TklrmBTb7/view?usp=drive_link)
2. **Set environment variable**: `export TELNET_DATADIR=/path/to/data`
3. **Configure CDS API** for ERA5 downloads: [How to use CDS API](https://cds.climate.copernicus.eu/how-to-api)

## Configuration

Edit `config/default.yaml` to customize:

```yaml
region:
  name: "maranhao"
  lat_min: -11.0
  lat_max: -1.0
  lon_min: -48.0
  lon_max: -41.5

model:
  epochs: 200
  nfeats: 10
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA (optional, for GPU training)

## License

See [LICENSE](LICENSE) file.
