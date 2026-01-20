import os
import h5py
import torch
import xarray as xr
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Dict, List
from random import choices, sample
from torch.utils.data import DataLoader
from models.model import TelNet
from utilities.plots import plot_obs_ypred_maps
from utils import CreateDataset, DataManager, EvalMetrics, make_dir, preprocess_num_models
from utils import DEVICE, exp_data_pt_dir, exp_data_dir, exp_results_dir
from utils import month2onehot, set_seed, compute_anomalies, scalar2categ

def load_metric_array(seed_n, metric, result_dir):
 
    checkpoint_file = f'{result_dir}/{metric}_{seed_n}.h5'
    if os.path.exists(checkpoint_file):
        # print ('Restarting the search from the last checkpoint ...', end = "\r")
        with h5py.File(checkpoint_file, 'r') as hf:
            arr = hf['dataset'][()]
    else:
        arr = None
   
    return arr

def save_metric_array(arr, seed_n, metric, result_dir):
    
    checkpoint_file = f'{result_dir}/{metric}_{seed_n}.h5'
    with h5py.File(checkpoint_file, 'w') as hf:
        hf.create_dataset('dataset', data=arr)

def save_varsel_wgts(Ydm, VSweights, index_list, outdir, fname):

    dims_vs = ['time', 'lead', 'indices']
    dvars = {'VSweights': (dims_vs, VSweights)}
    coords = {'time': (['time'], Ydm['test']['time'].values),
              'lead': (['lead'], Ydm['test']['time_seq'].values),
              'indices': (['indices'], np.append(['ylag'], index_list))}   
    ds = xr.Dataset(data_vars=dvars, coords=coords)
    ds.to_netcdf(f'{outdir}/{fname}.nc')

    return ds

def preprocess_data(
        Xdm: Dict[str, DataManager], 
        Ydm: DataManager, 
        train_yrs: np.ndarray, 
        val_yrs: np.ndarray, 
        test_yrs: np.ndarray,
        time_steps: int = 2,
        lead: int = 6,
        seed: int = 0
    ):
    
    print ('Preprocessing data ...')
    stat_yrs = np.asarray([i for i in train_yrs if i>=1971 and i<=2020])
    Xdm['auto'].monthly2seasonal('var', 'sum', True)
    Xdm['auto'].compute_statistics(stat_yrs, ['mean', 'std'], 'var')
    Xdm['auto'].to_anomalies('var', stdized=True)
    Xdm['auto'].apply_detrend()
    Xdm['auto'].add_seq_dim('var', time_steps)
    Xdm['auto'].replace_nan('var', -999.)

    # Xdm['cov'].compute_statistics(stat_yrs, ['mean', 'std'])
    # Xdm['cov'].to_anomalies('var', stdized=True)
    Xdm['cov'].apply_detrend()
    Xdm['cov'].add_seq_dim('var', time_steps)
    Xdm['cov'].replace_nan('var', -999.)
    
    Ydm.monthly2seasonal('var', 'sum', True)
    Ydm.compute_statistics(stat_yrs, ['mean', 'std'], 'var') 
    Ydm.to_anomalies('var', stdized=True)
    Ydm.apply_detrend()
    Ydm.compute_statistics(stat_yrs, ['terciles'], 'var')
    Ydm.add_seq_dim('var', lead, 'lead')
    Ydm.replace_nan('var', -999.)

    time_range = pd.date_range(Xdm['auto']['var'].time.values[0], 
                               Ydm['var'].time.values[-2], freq='MS')
    X_train_samples = [j.to_datetime64() for i in train_yrs for j in time_range if j.year == i]
    Y_train_samples = [(j+pd.DateOffset(months=1)).to_datetime64() for i in train_yrs for j in time_range if j.year == i]
    X_val_samples = [j.to_datetime64() for i in val_yrs for j in time_range if j.year == i]
    Y_val_samples = [(j+pd.DateOffset(months=1)).to_datetime64() for i in val_yrs for j in time_range if j.year == i]
    X_test_samples = [j.to_datetime64() for i in test_yrs for j in time_range if j.year == i]
    Y_test_samples = [(j+pd.DateOffset(months=1)).to_datetime64() for i in test_yrs for j in time_range if j.year == i]
    
    # Making sure no season is in both train and val
    X_train_samples = [i for i in X_train_samples 
                       if not np.isin(pd.date_range(start=i, periods=lead, freq='MS'), X_val_samples).any()
                          and
                          not np.isin(pd.date_range(end=i, periods=lead, freq='MS'), X_val_samples).any()]
    Y_train_samples = [i for i in Y_train_samples 
                       if not np.isin(pd.date_range(start=i, periods=lead, freq='MS'), Y_val_samples).any()
                          and
                          not np.isin(pd.date_range(end=i, periods=lead, freq='MS'), Y_val_samples).any()]

    Xdm['auto'].create_subsamples(['train', 'val', 'test'], [X_train_samples, X_val_samples, X_test_samples], 'var')
    Xdm['cov'].create_subsamples(['train', 'val', 'test'], [X_train_samples, X_val_samples, X_test_samples], 'var')
    Ydm.create_subsamples(['train', 'val', 'test'], [Y_train_samples, Y_val_samples, Y_test_samples], 'var')

    Xdm = {'auto': Xdm['auto'], 'cov': Xdm['cov']}

    return Xdm, Ydm

def split_sample():

    train_samples = np.arange(1941, 1993)
    val_samples = np.arange(1993, 2003)
    test_samples = np.arange(2003, 2023)
    test_subsamples = choices(test_samples, k=len(test_samples))  # Bootstrap sampling for test years
        
    return train_samples, val_samples, test_subsamples

def training(X: Dict[str, DataManager], 
             Y: DataManager, 
             model_config:list):
    
    nunits, drop, weight_scale, epochs, lr, clip, batch_size, nfeats, time_steps, lead, nmembs = model_config
    B = Y['train'].shape[0]
    H = X['auto']['train'].shape[-2]
    W = X['auto']['train'].shape[-1]
    I = nfeats
    D = nunits
    T = time_steps
    L = lead
    
    lats2d = Y.lat[:, None].repeat(W, 1)
    lat_wgts = torch.as_tensor(np.cos(np.deg2rad(lats2d)), device=DEVICE)
    lat_wgts = lat_wgts / lat_wgts.mean()
    lat_wgts = lat_wgts[None, None].tile((1, L, 1, 1))

    Xtrain_static = month2onehot(Y['train']['time.month'].values)
    Xtrain = [X['auto']['train'].values, X['cov']['train'].values, Xtrain_static]
    Ytrain = Y['train'].values
    train_dataset = CreateDataset(Xtrain, Ytrain, DEVICE)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    Xval_static = month2onehot(Y['val']['time.month'].values)
    Xval = [X['auto']['val'].values, X['cov']['val'].values, Xval_static]
    Yval = Y['val'].values
    val_dataset = CreateDataset(Xval, Yval, DEVICE)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    telnet = TelNet(nmembs, H, W, I, D, T, L, drop, weight_scale).to(DEVICE)
    if os.path.exists(f'{exp_data_pt_dir}/telnet.pt'):
        print ('Loading pre-trained model ...')
        telnet.load_state_dict(torch.load(f'{exp_data_pt_dir}/telnet.pt', weights_only=True))
    else:
        print ('Training model ...')
        telnet.train_model(train_dataloader, epochs, clip, lr=lr, val_dataloader=val_dataloader, lat_wgts=lat_wgts)
        torch.save(telnet.state_dict(), f'{exp_data_pt_dir}/telnet.pt')

    return telnet

def inference(X: Dict[str, DataManager], 
              Y: DataManager,
              telnet: nn.ModuleList,
              dset_name: str='val'):

    Xstatic_val = month2onehot(Y[dset_name]['time.month'].values)
    Xval = [
        torch.as_tensor(X['auto'][dset_name].values.astype(np.float32), device=DEVICE), 
        torch.as_tensor(X['cov'][dset_name].values.astype(np.float32), device=DEVICE),
        torch.as_tensor(Xstatic_val.astype(np.float32), device=DEVICE)
    ]
    xmask = torch.where(Xval[0][0:1, 0:1] == -999., 0., 1.).to(DEVICE)
    ypred, wgts = telnet.inference(Xval, xmask)
    Ypred = ypred.detach().cpu().numpy()
    Wgts = wgts.detach().cpu().numpy()

    return Ypred, Wgts

def evaluate_val(Y: DataManager, 
                 Ypred: np.ndarray,
                 init_months: list=[1, 4, 7, 10],
                 lead: int=6,):
    
    RPS = np.full((4, 6), np.nan)

    for n, i in enumerate(init_months):
        idcs = np.where((Y['val']['time.month'].values == i))[0]
        Yval_i = Y['val'][idcs]
        time_axis = np.concatenate([pd.date_range(j, periods=lead, freq='MS') for j in Yval_i.time.values])
        Yval_i = Yval_i.stack(time_stacked=('time', 'time_seq'), create_index=False).drop(('time', 'time_seq')).squeeze().rename(time_stacked='time').assign_coords({'time': time_axis}).transpose('time', 'lat', 'lon')
        Yval_i = xr.where(Yval_i==-999., np.nan, Yval_i)
        years = Yval_i[0::lead]['time.year'].values

        Ypred_i = Ypred[idcs].reshape(-1, Ypred.shape[-3], Ypred.shape[-2], Ypred.shape[-1])
        Yq33 = Y.q33.sel(time=time_axis).values
        Yq66 = Y.q66.sel(time=time_axis).values
        Ymn = Y.mn.sel(time=time_axis).values
        Ystd = Y.std.sel(time=time_axis).values
        
        Yval_i_categ, _, _ = scalar2categ(Yval_i.values, 3, 'one-hot', Yq33, Yq66)
        Ypred_i_probs, _, _ = scalar2categ(Ypred_i, 3, 'one-hot', Yq33, Yq66, count=True)

        for l in range(3, lead):
            RPS[n, l] = EvalMetrics.RPS(Yval_i_categ[l::lead].transpose(1, 0, 2, 3), Ypred_i_probs[l::lead].transpose(1, 0, 2, 3), ax=np.s_[0, 1, 2])

    return np.nanmean(RPS)

def evaluate_test(Y: DataManager, 
                  Ypred: np.ndarray,
                  baseline_models: dict,
                  init_months: list=[1, 4, 7, 10],
                  lead: int=6,
                  seed_n: int=0,
                  varsel_wgts: np.ndarray=None,
                  var_names: List[str]=None,
                  plot_ypred: bool=False):
    
    nyrs = Y['test'].shape[0]//12
    nvalid_points = np.where(Y['test'].values[0, 0].reshape(-1)!=-999.)[0].shape[0]
    nmembs = Ypred.shape[2]
    x, y = np.meshgrid(Y.lon, Y.lat)
    models_names = ['TelNet', 'CCSM4', 'CanCM4i', 'GEM-NEMO', 'GFDL', 'SEAS5']
    
    RMSE = np.full((len(baseline_models[11])+1, 4, 6, len(Y.lat), len(Y.lon)), np.nan)
    RPS = np.full((len(baseline_models[11]), 4, 6, len(Y.lat), len(Y.lon)), np.nan)
    # Rank histogram
    ranks = np.full((len(baseline_models[11]), 4, 6, nyrs*nvalid_points), np.nan)
    ranks_ext = np.full((len(baseline_models[11]), 4, 6, nyrs*nvalid_points), np.nan)
    # Reliability and sharpness diagrams
    ncategs = 3
    # bins = np.linspace(0., 1., 11)
    bins = np.linspace(0., 1., 6)
    obs_freq = np.full((len(baseline_models[11]), 4, 6, ncategs, len(bins)-1), np.nan)
    prob_avg = np.full((len(baseline_models[11]), 4, 6, ncategs, len(bins)-1), np.nan)
    pred_marginal = np.full((len(baseline_models[11]), 4, 6, ncategs, len(bins)-1), np.nan)
    rel = np.full((len(baseline_models[11]), 4, 6, ncategs), np.nan)
    res = np.full((len(baseline_models[11]), 4, 6, ncategs), np.nan)

    for n, i in enumerate(init_months):
        baseline_models_i = baseline_models[i+1]
        idcs = np.where((Y['test']['time.month'].values == i))[0]
        Yval_i = Y['test'][idcs]
        time_axis = np.concatenate([pd.date_range(j, periods=lead, freq='MS') for j in Yval_i.time.values])
        Yval_i = Yval_i.stack(time_stacked=('time', 'time_seq'), create_index=False).drop(('time', 'time_seq')).squeeze().rename(time_stacked='time').assign_coords({'time': time_axis}).transpose('time', 'lat', 'lon')
        Yval_i = xr.where(Yval_i==-999., np.nan, Yval_i)
        years = Yval_i[0::lead]['time.year'].values

        vsel_wgts_i = varsel_wgts[idcs].reshape(-1, varsel_wgts.shape[-1])

        Ypred_i = Ypred[idcs].reshape(-1, Ypred.shape[-3], Ypred.shape[-2], Ypred.shape[-1])
        Yq33 = Y.q33.sel(time=time_axis).values
        Yq66 = Y.q66.sel(time=time_axis).values
        Ymn = Y.mn.sel(time=time_axis).values
        Ystd = Y.std.sel(time=time_axis).values
        
        Yval_i_categ, _, _ = scalar2categ(Yval_i.values, ncategs, 'one-hot', Yq33, Yq66)
        Ypred_i_probs, _, _ = scalar2categ(Ypred_i, ncategs, 'one-hot', Yq33, Yq66, count=True)
        Yval_i_total = compute_anomalies(Yval_i.values, Ymn, Ystd, reverse=True)
        Ypred_i_total = compute_anomalies(Ypred_i, np.tile(Ymn[:, None], (1, nmembs, 1, 1)), np.tile(Ystd[:, None], (1, nmembs, 1, 1)), reverse=True)
        for l in range(lead):
            rmse_i_l = EvalMetrics.RMSE(Yval_i.values[l::lead], Ypred_i.mean(1)[l::lead])  # H, W
            RMSE[0, n, l] = rmse_i_l
            rps_i_l = EvalMetrics.RPS(Yval_i_categ[l::lead].transpose(1, 0, 2, 3), Ypred_i_probs[l::lead].transpose(1, 0, 2, 3))  # H, W
            RPS[0, n, l] = rps_i_l
            ranks_i_l = EvalMetrics.obs_rank(Yval_i.values[l::lead], Ypred_i[l::lead])
            ranks[0, n, l] = ranks_i_l
            ranks_ext_i_l = EvalMetrics.obs_rank(Yval_i.values[l::lead], Ypred_i[l::lead], 1)
            ranks_ext[0, n, l] = ranks_ext_i_l
            obs_freq_i_l, prob_avg_i_l, pred_marginal_i_l, rel_i_l, res_i_l = EvalMetrics.calibration_refinement_functions(Yval_i_categ[l::lead].transpose(1, 0, 2, 3), 
                                                                                                                           Ypred_i_probs[l::lead].transpose(1, 0, 2, 3), 
                                                                                                                           bins)
            obs_freq[0, n, l] = obs_freq_i_l
            prob_avg[0, n, l] = prob_avg_i_l
            pred_marginal[0, n, l] = pred_marginal_i_l
            rel[0, n, l] = rel_i_l
            res[0, n, l] = res_i_l           

        for m, model in enumerate(baseline_models_i):
            model_time = [i for i in time_axis if i in model.time.values]
            Ypred_num_i = model.sel(time=model_time) #.values

            nmembs_num = Ypred_num_i.shape[1]
            Ymn = Y.mn.sel(time=model_time).values
            Ystd = Y.std.sel(time=model_time).values
            Ypred_num_anom_i = (Ypred_num_i.values - np.tile(Ymn[:, None], (1, nmembs_num, 1, 1)))/np.tile(Ystd[:, None], (1, nmembs_num, 1, 1))
            
            if nmembs_num > 1:
                Yq33 = Y.q33.sel(time=model_time).values
                Yq66 = Y.q66.sel(time=model_time).values
                Ypred_num_i_probs, _, _ = scalar2categ(Ypred_num_anom_i, 3, 'one-hot', Yq33, Yq66, count=True)
            for l in range(3, lead):
                rmse_i_l = EvalMetrics.RMSE(Yval_i.values[l::lead], Ypred_num_anom_i.mean(1)[l-3::3])
                RMSE[m+1, n, l] = rmse_i_l
                rps_i_l = EvalMetrics.RPS(Yval_i_categ[l::lead].transpose(1, 0, 2, 3), Ypred_num_i_probs[l-3::3].transpose(1, 0, 2, 3))
                if nmembs_num > 1:
                    RPS[m+1, n, l] = rps_i_l
                    ranks_i_l = EvalMetrics.obs_rank(Yval_i.values[l::lead], Ypred_num_anom_i[l-3::3])
                    ranks[m+1, n, l] = ranks_i_l
                    ranks_ext_i_l = EvalMetrics.obs_rank(Yval_i.values[l::lead], Ypred_num_anom_i[l-3::3], 1)
                    ranks_ext[m+1, n, l] = ranks_ext_i_l
                    obs_freq_i_l, prob_avg_i_l, pred_marginal_i_l, rel_i_l, res_i_l = EvalMetrics.calibration_refinement_functions(Yval_i_categ[l::lead].transpose(1, 0, 2, 3), 
                                                                                                                                Ypred_num_i_probs[l-3::3].transpose(1, 0, 2, 3), 
                                                                                                                                bins)
                    obs_freq[m+1, n, l] = obs_freq_i_l
                    prob_avg[m+1, n, l] = prob_avg_i_l
                    pred_marginal[m+1, n, l] = pred_marginal_i_l
                    rel[m+1, n, l] = rel_i_l
                    res[m+1, n, l] = res_i_l
            
        if plot_ypred:
            if i == 10:
                title = 'Nov-DJF'
            elif i == 1:
                title = 'Feb-MAM'
            elif i == 4:
                title = 'May-JJA'
            elif i == 7:
                title = 'Aug-SON'
            result_dir = f'{exp_results_dir}/test'
            make_dir(f'{result_dir}/forecasts_{seed_n}')
            for n, yr in enumerate(years):
                plot_obs_ypred_maps(x, y, [yr], 
                                Yval_i_total[3::lead][n:n+1], Ypred_i_total.mean(1)[3::lead][n:n+1], 
                                Yval_i.values[3::lead][n:n+1], Ypred_i.mean(1)[3::lead][n:n+1], 
                                Yval_i_categ[3::lead][n:n+1], Ypred_i_probs[3::lead][n:n+1],
                                vsel_wgts_i[3::lead][n], var_names, '', 
                                f'forecast_{yr}_{title}.png', f'{result_dir}/forecasts_{seed_n}', (13, 7),
                                levels_totals=[0, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400])
        
    RMSESS = np.concatenate([((1-(np.nanmean(RMSE[0], axis=np.s_[2, 3])/np.nanmean(RMSE[i], axis=np.s_[2, 3])))*100)[None] for i in range(1, RMSE.shape[0])], 0)  # nmodels, ninits, nleads
    RPSS = np.concatenate([((1-(np.nanmean(RPS[0], axis=np.s_[2, 3])/np.nanmean(RPS[i], axis=np.s_[2, 3])))*100)[None] for i in range(1, RPS.shape[0])], 0)  # nmodels, ninits, nleads

    return RMSE, RPS, ranks, ranks_ext, obs_freq, prob_avg, pred_marginal, rel, res, RMSESS, RPSS

def main(arguments):

    seed_n, X, Y, model_config, model_predictors, seeds, baseline_models, init_months = arguments
    checkpoints_dir = f'{exp_data_dir}/checkpoints_model'
    metric_names = ['Yval_RPS', 'RMSE', 'RPS', 'ranks', 'ranks_ext', 'obs_freq', 'prob_avg', 'pred_marginal', 'rel', 'res', 'RMSESS', 'RPSS']
    metrics = [None]*len(metric_names)
    plot_ypred = False # Force false to avoid Colab crashes
    # plot_ypred = True if seed_n % 10 == 0 else False
    if os.path.exists(f'{checkpoints_dir}/varsel_{seed_n}.nc'):
        vs_ds = xr.open_dataset(f'{checkpoints_dir}/varsel_{seed_n}.nc')
    else:
        vs_ds = None
    for i, metric in enumerate(metrics):
        metrics[i] = load_metric_array(seed_n, metric_names[i], checkpoints_dir)
    if all([metric is not None for metric in metrics]):
        print (f'All metrics already computed for seed {seed_n}, Yval_RPS={metrics[0]:.4f}')
        Yval_RPS, RMSE, RPS, ranks, ranks_ext, obs_freq, prob_avg, pred_marginal, rel, res, RMSESS, RPSS = metrics
        return Yval_RPS, RMSE, RPS, ranks, ranks_ext, obs_freq, prob_avg, pred_marginal, rel, res, RMSESS, RPSS, vs_ds['VSweights']

    seed_pos = np.argwhere(seeds == seed_n).flatten()[0]
    print (f'Running sample {seed_pos+1}/{len(seeds)} ...')
    set_seed(seed_n)
    
    train_yrs, val_yrs, test_yrs = split_sample()
    
    time_steps = model_config[-3]
    lead = model_config[-2]
    nmembs = model_config[-1]
    Xdm, Ydm = preprocess_data(X, Y, train_yrs, val_yrs, test_yrs, time_steps, lead, seed_n)
    telnet = training(Xdm, Ydm, model_config)
    Ypred_val, _ = inference(Xdm, Ydm, telnet, 'val')
    Yval_RPS = evaluate_val(Ydm, Ypred_val, init_months, lead)
    Ypred_test, Wgts = inference(Xdm, Ydm, telnet, 'test')
    
    for i in init_months:
        dyn_totals = preprocess_num_models(baseline_models[i+1]['dyn'], test_yrs, i+1, Ydm['mn'], Ydm['std'])
        dl_totals = preprocess_num_models(baseline_models[i+1]['dl'], test_yrs, i+1,  Ydm['mn'], Ydm['std'])
        dyn_dl_totals = dyn_totals + dl_totals
        baseline_models[i+1] = dyn_dl_totals
    
    RMSE, RPS, ranks, ranks_ext, obs_freq, prob_avg, pred_marginal, rel, res, RMSESS, RPSS = evaluate_test(Ydm, Ypred_test, baseline_models, init_months, lead, seed_n, Wgts, np.append(['ylag'], model_predictors), plot_ypred)
    metrics = [Yval_RPS, RMSE, RPS, ranks, ranks_ext, obs_freq, prob_avg, pred_marginal, rel, res, RMSESS, RPSS]
    
    for i, metric in enumerate(metrics):
        save_metric_array(metric, seed_n, metric_names[i], checkpoints_dir)
    vs_ds = save_varsel_wgts(Ydm, Wgts, model_predictors, checkpoints_dir, f'varsel_{seed_n}')

    del telnet
    torch.cuda.empty_cache()
    print (f'Finished sample {seed_pos+1}/{len(seeds)} with Yval_RPS={Yval_RPS:.4f} ...')
    return Yval_RPS, RMSE, RPS, ranks, ranks_ext, obs_freq, prob_avg, pred_marginal, rel, res, RMSESS, RPSS, vs_ds['VSweights']

if __name__ == '__main__':

    main()
