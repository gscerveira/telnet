from copy import deepcopy
import os
import torch
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict
from datetime import datetime
from models.model import TelNet
from utils import DEVICE, exp_data_dir, exp_data_pt_dir, exp_results_dir, DataManager, read_indices_data, shape2mask, prepare_X_data, month2onehot, compute_anomalies, scalar2categ
from utilities.plots import plot_ypred_maps


def read_era5_pcp(datadir, region_mask=None, mask_ocean=False, period=('1940-01-01', '2024-12-01')):

    if period[0] == '1940-01-01':
        df_var = xr.open_dataset(f'{datadir}/era5_pr_1940-present_preprocessed.nc').sel(time=slice(period[0], period[1]))
    elif period[0] == '2025-01-01':
        df_var = xr.open_dataset(f'{datadir}/era5_pr_2025-present_preprocessed.nc').sel(time=slice(period[0], period[1]))
    if region_mask is not None:
        mask = shape2mask(region_mask, df_var['lon'].values, df_var['lat'].values, 1.)
        df_var['pr'].values[:, ~mask] = np.nan
    if mask_ocean:
        lat = df_var['lat'].values
        lon = df_var['lon'].values
        mask = xr.open_dataset(f'{datadir}/era5_land_sea_mask_1940-present_preprocessed.nc').sel(lat=lat, lon=lon).isel(time=0)
        df_var = df_var.where(np.tile(mask['lsm'].values, (df_var.time.size, 1, 1)) > 0.5, np.nan)

    return df_var

def read_data(forecast_date: str):
    root_datadir = os.getenv('TELNET_DATADIR')
    era5_dir = os.path.join(root_datadir, 'era5')
    
    hist_indices = read_indices_data('1941-01-01', '2023-12-01', root_datadir, institute='_1941-2024')
    present_indices = read_indices_data('2025-01-01', forecast_date, root_datadir, institute='_2025-present')
    indices = pd.concat([hist_indices, present_indices])

    hist_pcp = read_era5_pcp(era5_dir, period=('1940-01-01', '2024-12-01'), mask_ocean=True)
    present_pcp = read_era5_pcp(era5_dir, period=('2025-01-01', forecast_date), mask_ocean=True)
    pcp = xr.concat([hist_pcp, present_pcp], dim='time')

    cov_date_s = ('1941-01-01', forecast_date)
    auto_date_s = ('1940-12-01', forecast_date)

    X = {'auto': deepcopy(pcp['pr']), 'cov': deepcopy(indices)}
    Xdm = prepare_X_data(X, auto_date_s, cov_date_s, 
                       cov_bounds=((None, None), (None, None)), 
                       auto_bounds=((None, None), (None, None)))

    return Xdm

def preprocess_data(
        Xdm: Dict[str, DataManager], 
        forecast_date: np.ndarray,
        time_steps: int = 2,
    ):
    
    print ('Preprocessing data ...')
    stat_yrs = np.arange(1971, 1993)
    Xdm['auto'].monthly2seasonal('var', 'sum', True)
    Xdm['auto'].compute_statistics(stat_yrs, ['mean', 'std'], 'var')
    Xdm['auto'].to_anomalies('var', stdized=True)
    Xdm['auto'].apply_detrend()
    Xdm['auto'].compute_statistics(stat_yrs, ['terciles'], 'var')
    Xdm['auto'].add_seq_dim('var', time_steps)
    Xdm['auto'].replace_nan('var', -999.)

    # Xdm['cov'].compute_statistics(stat_yrs, ['mean', 'std'])
    # Xdm['cov'].to_anomalies('var', stdized=True)
    Xdm['cov'].apply_detrend()
    Xdm['cov'].add_seq_dim('var', time_steps)
    Xdm['cov'].replace_nan('var', -999.)
    
    Xdm['auto'].create_subsamples(['forecast'], [np.datetime64(forecast_date)-pd.DateOffset(months=1)], 'var')
    Xdm['cov'].create_subsamples(['forecast'], [np.datetime64(forecast_date)-pd.DateOffset(months=1)], 'var')

    Xdm = {'auto': Xdm['auto'], 'cov': Xdm['cov']}

    return Xdm

def generate_forecast(X, model_config):

    nunits, drop, weight_scale, epochs, lr, clip, batch_size, nfeats, time_steps, lead, nmembs = model_config
    B = X['auto']['forecast'].shape[0]
    H = X['auto']['forecast'].shape[-2]
    W = X['auto']['forecast'].shape[-1]
    I = nfeats
    D = nunits
    T = time_steps
    L = lead

    telnet = TelNet(nmembs, H, W, I, D, T, L, drop, weight_scale).to(DEVICE)
    if os.path.exists(f'{exp_data_pt_dir}/telnet.pt'):
        print ('Loading pre-trained model ...')
        telnet.load_state_dict(torch.load(f'{exp_data_pt_dir}/telnet.pt', weights_only=True))
    
    Xstatic_val = month2onehot(np.array([(X['auto']['forecast']['time'].values + pd.DateOffset(months=1)).month]))
    Xfore = [
        torch.as_tensor(X['auto']['forecast'].values[None].astype(np.float32), device=DEVICE), 
        torch.as_tensor(X['cov']['forecast'].values[None].astype(np.float32), device=DEVICE),
        torch.as_tensor(Xstatic_val.astype(np.float32), device=DEVICE)
    ]
    xmask = torch.where(Xfore[0][0:1, 0:1] == -999., 0., 1.).to(DEVICE)
    telnet.eval()
    with torch.no_grad():
        Ypred, Wgts = telnet.inference(Xfore, xmask)  # Ypred=[B, L, N, H, W], Wgts=[B, L, I+1]

    Ypred = Ypred.detach().cpu().numpy()[:, 2:, :, :]  # Keep leads 2, 3, 4, 5
    Wgts = Wgts.detach().cpu().numpy()[:, 2:, :]  # Keep leads 2, 3, 4, 5
    
    return Ypred, Wgts

def main(forecast_date: str, config_n: int, figsize: tuple):

    init_time = datetime.now()
    forecast_dir = os.path.join(exp_results_dir, 'forecasts')
    os.makedirs(forecast_dir, exist_ok=True)
    
    print (f'Reading search matrix from {exp_data_dir}/search_matrix.csv')
    df_search = pd.read_csv(os.path.join(exp_data_dir, 'search_matrix.csv'), index_col=0)
    model_config = df_search.iloc[config_n]
    config_n = int(model_config.name)
    model_config = model_config.values
    model_config = [int(x) if x % 1 == 0 else float(x) for x in model_config]
    nfeats = model_config[7]
    time_steps = model_config[8]
    nmembs = model_config[10]
    lead = model_config[9]
    model_predictors = np.loadtxt(os.path.join(f'{exp_data_dir}/models/', f'final_feats.txt'), dtype=str, delimiter=' ')[:nfeats]
    X = read_data(forecast_date)
    X['cov']['var'] = X['cov']['var'].sel(indices=model_predictors)
    Xdm = preprocess_data(X, forecast_date, time_steps)
    Ypred, Wgts = generate_forecast(Xdm, model_config)

    x, y = np.meshgrid(Xdm['auto'].lon, Xdm['auto'].lat)

    
    forecast_range = pd.date_range(start=forecast_date, periods=4, freq='MS') + pd.DateOffset(months=1) # Leads 2, 3, 4, 5
    forecast_months = forecast_range.month.values
    init_months_str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    xlabels = {'Jan': ['JFM', 'FMA', 'MAM', 'AMJ'],
               'Feb': ['FMA', 'MAM', 'AMJ', 'MJJ'],
               'Mar': ['MAM', 'AMJ', 'MJJ', 'JJA'],
               'Apr': ['AMJ', 'MJJ', 'JJA', 'JAS'],
               'May': ['MJJ', 'JJA', 'JAS', 'ASO'],
               'Jun': ['JJA', 'JAS', 'ASO', 'SON'],
               'Jul': ['JAS', 'ASO', 'SON', 'OND'],
               'Aug': ['ASO', 'SON', 'OND', 'NDJ'],
               'Sep': ['SON', 'OND', 'NDJ', 'DJF'],
               'Oct': ['OND', 'NDJ', 'DJF', 'JFM'],
               'Nov': ['NDJ', 'DJF', 'JFM', 'FMA'],
               'Dec': ['DJF', 'JFM', 'FMA', 'MAM']}
    
    ncategs = 3
    tar_dir = os.path.join(forecast_dir, f'init_{forecast_range[0].strftime("%Y%m")}')
    os.makedirs(tar_dir, exist_ok=True)
    for n, mon in enumerate(forecast_months):
        
        mn = Xdm['auto'].mn.sel(time=f'1990-{mon}-01').values[None]
        std = Xdm['auto'].std.sel(time=f'1990-{mon}-01').values[None]
        q33 = Xdm['auto'].q33.sel(time=f'1990-{mon}-01').values[None]
        q66 = Xdm['auto'].q66.sel(time=f'1990-{mon}-01').values[None]

        Ypred_i = Ypred[:, n, :, :, :]  # (B, N, H, W)
        Wgts_i = Wgts[0, n, :]  # (I+1)

        Ypred_i_probs, _, _ = scalar2categ(Ypred_i, ncategs, 'one-hot', q33, q66, count=True)
        Ypred_i_total = compute_anomalies(Ypred_i, np.tile(mn[:, None], (1, nmembs, 1, 1)), np.tile(std[:, None], (1, nmembs, 1, 1)), reverse=True)

        title = f'TelNet Forecast - Init: {init_months_str[forecast_months[0]]} - Target: {xlabels[init_months_str[forecast_months[0]]][n]}'
        filename = f'telnet_forecast_{init_months_str[forecast_months[0]]}_lead{n}.png'
        var_names = ['Ypred'] + list(model_predictors[:nfeats])
        plot_ypred_maps(x, y, forecast_date, Ypred_i_total.mean(1), Ypred_i.mean(1), Ypred_i_probs, Wgts_i, var_names, title, filename, tar_dir, figsize=figsize)

    end_time = datetime.now()
    print ('Total forecast generation time: ', end_time - init_time)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate TelNet Forecast')
    parser.add_argument('-fdate', '--forecast_date', help='Forecast initialization date in YYYY-MM format', required=True)
    parser.add_argument('-c', '--config', help='Configuration number', required=True)

    args = parser.parse_args()
    forecast_date = f'{args.forecast_date[:4]}-{args.forecast_date[4:6]}-01'
    config_n = int(args.config)

    # CHANGE FORECAST FIGURE SIZE HERE IF NEEDED
    figsize = (16, 10)  # (x, y)

    main(forecast_date, config_n, figsize)