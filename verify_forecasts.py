"""
Verify TelNet forecasts against observed ERA5 precipitation.
Computes skill scores and generates reliability diagrams.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_observations(obs_path):
    """Load observed precipitation data."""
    return xr.open_dataset(obs_path)


def load_forecast_ensemble(forecast_path):
    """Load forecast ensemble data."""
    return xr.open_dataset(forecast_path)


def compute_tercile_categories(data, climatology):
    """
    Compute tercile categories (below/normal/above) based on climatology.

    Returns integer array: 0=below, 1=normal, 2=above
    """
    lower_tercile = climatology.quantile(1/3, dim='time')
    upper_tercile = climatology.quantile(2/3, dim='time')

    categories = xr.where(data < lower_tercile, 0,
                 xr.where(data > upper_tercile, 2, 1))
    return categories


def compute_tercile_probabilities(ensemble):
    """
    Compute tercile probabilities from ensemble members.

    Parameters
    ----------
    ensemble : xr.DataArray
        Ensemble forecast with 'member' dimension

    Returns
    -------
    xr.Dataset
        Dataset with prob_below, prob_normal, prob_above variables
    """
    climatology = ensemble.mean(dim='member')
    lower_tercile = climatology.quantile(1/3)
    upper_tercile = climatology.quantile(2/3)

    prob_below = (ensemble < lower_tercile).mean(dim='member')
    prob_above = (ensemble > upper_tercile).mean(dim='member')
    prob_normal = 1 - prob_below - prob_above

    return xr.Dataset({
        'prob_below': prob_below,
        'prob_normal': prob_normal,
        'prob_above': prob_above
    })


def compute_rps(forecast_probs, observed_category):
    """
    Compute Ranked Probability Score.

    Parameters
    ----------
    forecast_probs : xr.Dataset
        Dataset with prob_below, prob_normal, prob_above
    observed_category : xr.DataArray
        Observed category (0, 1, or 2)

    Returns
    -------
    xr.DataArray
        RPS values (lower is better, 0 is perfect)
    """
    # Cumulative probabilities
    cum_forecast = xr.concat([
        forecast_probs['prob_below'],
        forecast_probs['prob_below'] + forecast_probs['prob_normal'],
        xr.ones_like(forecast_probs['prob_below'])
    ], dim='category')

    # Cumulative observed (step function)
    cum_observed = xr.concat([
        (observed_category >= 1).astype(float),
        (observed_category >= 2).astype(float),
        xr.ones_like(observed_category, dtype=float)
    ], dim='category')

    rps = ((cum_forecast - cum_observed) ** 2).sum(dim='category') / 2
    return rps


def compute_rpss(rps, rps_climatology):
    """
    Compute Ranked Probability Skill Score.

    RPSS = 1 - RPS/RPS_clim

    RPSS > 0: better than climatology
    RPSS = 0: same as climatology
    RPSS < 0: worse than climatology
    """
    return 1 - rps / rps_climatology


def compute_climatology_rps():
    """RPS for climatological forecast (1/3, 1/3, 1/3)."""
    return 2/9  # Analytical result for tercile climatology


def reliability_diagram(forecast_probs, observed_binary, n_bins=10, ax=None):
    """
    Create reliability diagram for probabilistic forecasts.

    Parameters
    ----------
    forecast_probs : array-like
        Forecast probabilities [0, 1]
    observed_binary : array-like
        Binary observations (0 or 1)
    n_bins : int
        Number of probability bins
    ax : matplotlib axis, optional

    Returns
    -------
    dict
        Reliability statistics
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    forecast_probs = np.array(forecast_probs).flatten()
    observed_binary = np.array(observed_binary).flatten()

    # Remove NaN
    valid = ~(np.isnan(forecast_probs) | np.isnan(observed_binary))
    forecast_probs = forecast_probs[valid]
    observed_binary = observed_binary[valid]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    observed_freq = []
    forecast_freq = []
    counts = []

    for i in range(n_bins):
        mask = (forecast_probs >= bin_edges[i]) & (forecast_probs < bin_edges[i+1])
        if mask.sum() > 0:
            observed_freq.append(observed_binary[mask].mean())
            forecast_freq.append(forecast_probs[mask].mean())
            counts.append(mask.sum())
        else:
            observed_freq.append(np.nan)
            forecast_freq.append(bin_centers[i])
            counts.append(0)

    # Plot
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect reliability')
    ax.plot(forecast_freq, observed_freq, 'bo-', label='Forecast')
    ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.1, color='gray')

    ax.set_xlabel('Forecast probability')
    ax.set_ylabel('Observed frequency')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_aspect('equal')

    return {
        'bin_centers': bin_centers,
        'forecast_freq': forecast_freq,
        'observed_freq': observed_freq,
        'counts': counts
    }


def verify_forecast(forecast_path, obs_path, output_dir, shapefile_dir=None):
    """
    Run full verification suite for a forecast.

    Parameters
    ----------
    forecast_path : str
        Path to forecast NetCDF (with ensemble members)
    obs_path : str
        Path to observed precipitation NetCDF
    output_dir : str
        Directory to save verification outputs
    shapefile_dir : str, optional
        If provided, mask to Maranhão
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    forecast = xr.open_dataset(forecast_path)
    obs = xr.open_dataset(obs_path)

    # Optionally mask to Maranhão
    if shapefile_dir:
        from extract_maranhao import mask_outside_maranhao
        forecast = mask_outside_maranhao(forecast, shapefile_dir)
        obs = mask_outside_maranhao(obs, shapefile_dir)

    print("Computing tercile probabilities...")
    # Assuming forecast has 'pr' variable with 'member' dimension
    probs = compute_tercile_probabilities(forecast['pr'])

    print("Computing observed categories...")
    # Use forecast climatology for tercile boundaries
    obs_category = compute_tercile_categories(obs['pr'], forecast['pr'].mean(dim='member'))

    print("Computing RPS...")
    rps = compute_rps(probs, obs_category)
    rps_clim = compute_climatology_rps()
    rpss = compute_rpss(rps, rps_clim)

    # Spatial mean statistics
    mean_rps = float(rps.mean())
    mean_rpss = float(rpss.mean())

    print(f"\nVerification Results:")
    print(f"  Mean RPS:  {mean_rps:.4f}")
    print(f"  Mean RPSS: {mean_rpss:.4f}")

    # Save statistics
    stats_df = pd.DataFrame({
        'metric': ['RPS', 'RPSS'],
        'value': [mean_rps, mean_rpss]
    })
    stats_df.to_csv(os.path.join(output_dir, 'skill_scores.csv'), index=False)

    # Create reliability diagrams
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (cat, name) in enumerate([('below', 'Below Normal'),
                                      ('normal', 'Normal'),
                                      ('above', 'Above Normal')]):
        prob_key = f'prob_{cat}'
        obs_binary = (obs_category == i).values
        reliability_diagram(probs[prob_key].values, obs_binary, ax=axes[i])
        axes[i].set_title(f'{name} Reliability')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reliability_diagrams.png'), dpi=150)
    plt.close()

    print(f"\nOutputs saved to {output_dir}")

    return {
        'rps': mean_rps,
        'rpss': mean_rpss,
        'rps_spatial': rps,
        'rpss_spatial': rpss
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Verify forecasts against observations')
    parser.add_argument('forecast', help='Forecast NetCDF file')
    parser.add_argument('observations', help='Observations NetCDF file')
    parser.add_argument('--output-dir', '-o', default='verification',
                        help='Output directory')
    parser.add_argument('--shapefile-dir', default=None,
                        help='Shapefile directory for Maranhão masking')

    args = parser.parse_args()

    verify_forecast(args.forecast, args.observations, args.output_dir, args.shapefile_dir)
