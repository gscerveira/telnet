import os
import torch
import numpy as np
import random
import inspect
import pandas as pd
import xarray as xr
import torch.nn as nn
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from math import ceil
from random import sample
from itertools import product
from copy import deepcopy
from typing import Dict, List, Union
from cartopy.feature import ShapelyFeature
from shapely import geometry
from calendar import monthrange
from scipy.signal import detrend
from torch.utils.data import Dataset


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print (DEVICE)

# Dir paths
currdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
utils_dir = f'{currdir}/utilities'
exp_data_dir = f'{currdir}/data'
if not os.path.exists(exp_data_dir):
    os.mkdir(exp_data_dir)
exp_data_pt_dir = f'{exp_data_dir}/models'
if not os.path.exists(exp_data_pt_dir):
    os.mkdir(exp_data_pt_dir)
exp_results_dir = f'{currdir}/results'
if not os.path.exists(exp_results_dir):
    os.mkdir(exp_results_dir)

## Shapefiles
shp_ce = shpreader.Reader(f'{utils_dir}/shape_ce/UFE250GC_SIR.shp')
shp_ce_feature = ShapelyFeature(shp_ce.geometries(), ccrs.PlateCarree())
for f in shp_ce.records():
    a = (f.geometry)
x_, y_ = a.exterior.coords.xy
points_ce = geometry.Polygon([(a, s) for a, s in zip(x_, y_)])

# Functions
def get_search_matrix():

    # Hyperparameters
    nunits = [1024, 512]  #  
    dropout = [0.1, 0.2]  # 
    weight_scale = [0.02]
    epochs = [10]  # 
    max_grad_norm = [0.01]  # 
    learning_rate = [1e-3, 1e-4]  # 
    batch_sizes = [1024, 1]
    nfeats = [4, 2]  # 
    time_steps = [2, 1]  #  
    lead = [6]
    nmembs = [20]

    search_matrix = list(product(nunits, dropout, weight_scale, epochs, learning_rate, max_grad_norm, batch_sizes, nfeats, time_steps, lead, nmembs))
    
    cols = ['nunits', 'dropout', 'weight_scale', 'epochs', 'learning_rate', 'clip', 'batch_size', 'nfeats', 'time_steps', 'lead', 'nmembs']
    df_search = pd.DataFrame(search_matrix, columns=cols)
    df_search.to_csv(f'{exp_data_dir}/search_matrix.csv', index=True)

    return search_matrix, df_search

def read_obs_data():
    root_datadir = os.getenv('TELNET_DATADIR')
    if root_datadir is None:
        raise ValueError('Environment variable TELNET_DATADIR is not set.')

    idcs_list = ['oni', 'atn-sst', 'ats-sst', 'atl-sst', 'iod', 'iobw', 'nao', 'pna', 'aao', 'ao']
    indices = read_indices_data('1941-01-01', '2023-12-01', root_datadir, idcs_list, '_1941-2024')

    # Load ERA5 from local preprocessed file
    print("Loading ERA5 precipitation from local file...")
    pcp = read_era5_data('pr', root_datadir, mask_ocean=False, period=('1940-01-01', '2024-01-01'))

    cov_date_s = ('1941-01-01', '2023-12-01')
    auto_date_s = ('1940-12-01', '2024-01-01')
    pred_date_s = ('1940-12-01', '2024-01-01')
    target_bounds = ((None, None), (None, None))

    X = {'auto': deepcopy(pcp['pr']), 'cov': deepcopy(indices)}
    Y = deepcopy(pcp['pr'])

    Xdm = prepare_X_data(X, auto_date_s, cov_date_s,
                         cov_bounds=((None, None), (None, None)),
                         auto_bounds=target_bounds)

    Ydm = prepare_Y_data(Y, pred_date_s, region_bounds=target_bounds)

    return Xdm, Ydm, idcs_list

def read_dl_models_data(init_month, Ylat, Ylon):

    months_str = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun', 
                  7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'}
    
    root_datadir = os.getenv('TELNET_DATADIR')
    if root_datadir is None:
        raise ValueError('Environment variable TELNET_DATADIR is not set.')
    dynmodel_dir = f'{root_datadir}/deeplearning_models_data/{months_str[init_month]}/'
    
    models = ['climax']
    models_totals = []
    for model in models:
        model_name = [i for i in os.listdir(dynmodel_dir) if i.startswith(f'pr_seasonal_{model}')][0]
        ds = xr.open_dataset(f'{dynmodel_dir}/{model_name}')
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        ds = ds.sortby(ds.lon)
        pcp = ds['Ypred'].reindex(lat=ds.lat[::-1])
        pcp = pcp.interp(lat=Ylat, lon=Ylon, method='linear')
        models_totals.append(pcp)
    return models_totals, models

def read_num_models_data(init_month, Ylat, Ylon):

    months_str = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun', 
                  7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'}
    
    root_datadir = os.getenv('TELNET_DATADIR')
    if root_datadir is None:
        raise ValueError('Environment variable TELNET_DATADIR is not set.')
    dynmodel_dir = f'{root_datadir}/numerical_models_data/{months_str[init_month]}/'
    
    models = ['cola-rsmas-ccsm4', 'cancm4i-ic3', 'gem5-nemo', 'gfdl-spear', 'seas5']
    models_totals = []
    for model in models:
        model_name = [i for i in os.listdir(dynmodel_dir) if i.startswith(f'pr_seasonal_{model}')][0]
        ds = xr.open_dataset(f'{dynmodel_dir}/{model_name}')
        pcp = ds['Ypred'].reindex(lat=ds.lat[::-1])
        pcp = pcp.interp(lat=Ylat, lon=Ylon, method='linear')
        models_totals.append(pcp)
    return models_totals, models

def preprocess_num_models(models_totals, test_yrs, init_month, Ymn_obs, Ystd_obs):

    models_totals_preprocessed = []
    for pcp in models_totals:
        pcp = compute_num_models_std_anom(pcp, test_yrs)
        pcp = flatten_lead_dim(pcp, init_month)
        pcp = compute_num_models_totals(pcp, Ymn_obs, Ystd_obs)
        models_totals_preprocessed.append(pcp)
    return models_totals_preprocessed

def flatten_lead_dim(darr, init_month):

    # +1 to center the seasonal window
    time_axis = [np.datetime64(f'{i}-{((init_month+ceil(j))%12)+1:02d}-01') 
                 for i in darr.time.values for j in darr.leads.values]

    darr = darr.stack({'time_flatten': ['time', 'leads']}).transpose('time_flatten', 'nmembs', 'lat', 'lon').drop(['time', 'leads'])
    darr = darr.rename({'time_flatten': 'time'})
    
    darr['time'] = time_axis
    
    return darr

def compute_num_models_std_anom(darr, test_yrs):
    """
    Computing statistics and anomalies through leave-one-out
    """
    yrs = darr.time.values
    # clim_yrs = np.array([i for i in stat_yrs if i not in test_yrs])
    Ypred_anom = []
    for i in yrs:
        clim_yrs = np.array([j for j in yrs if j != i])
        mn = darr.sel(time=clim_yrs).mean('nmembs').mean('time')
        std = darr.sel(time=clim_yrs).mean('nmembs').std('time')
        Ypred_anom.append(((darr.sel(time=i) - mn)/std))
        # Ypred_anom = ((darr - mn)/std)
    Ypred_anom = xr.concat(Ypred_anom, dim='time')

    return Ypred_anom

def compute_num_models_totals(darr, mn_obs, std_obs):

    darr*=std_obs.sel(time=darr.time.values)
    darr+=mn_obs.sel(time=darr.time.values)
    return darr

def read_indices_data(init_date, final_date, datadir, indices='all', institute='_noaa'):

    """
    Reads indices file, selects the indices given as input and slices it to the given period 
    """

    df = pd.read_csv(f'{datadir}/seasonal_climate_indices{institute}.txt', sep=' ', index_col=0, parse_dates=True)
    if indices == 'all':
        df = df.loc[(df.index >= init_date) & (df.index <= final_date)]
    else:
        df = df[indices].loc[(df.index >= init_date) & (df.index <= final_date)]

    return df

def read_era5_data(var, datadir=None, region_mask=None, mask_ocean=False, period=('1940-01-01', '2024-12-01')):
    """
    Read ERA5 data from local preprocessed files (downloaded via CDS API).
    """
    if datadir is None:
        datadir = os.getenv('TELNET_DATADIR')

    era5_dir = os.path.join(datadir, 'era5')

    # Load from local preprocessed file
    pr_file = os.path.join(era5_dir, 'era5_pr_1940-present_preprocessed.nc')
    print(f"  [ERA5] Loading precipitation from local file: {pr_file}")
    ds = xr.open_dataset(pr_file)

    # Slice to requested period
    ds = ds.sel(time=slice(period[0], period[1]))
    print(f"  [ERA5] Time range: {len(ds.time)} months")

    # Apply region mask if provided
    if region_mask is not None:
        print("  [ERA5] Applying region mask...")
        mask = shape2mask(region_mask, ds['lon'].values, ds['lat'].values, 1.)
        ds['pr'].values[:, ~mask] = np.nan

    print("  [ERA5] Data ready.")
    return ds

def shape2mask(poly_limis, x, y, buffer):

    s = geometry.Polygon(poly_limis).buffer(buffer)
    gridx, gridy = np.meshgrid(x, y)        
    gridx, gridy = gridx.flatten(), gridy.flatten()
    points = list(map(geometry.Point, np.vstack((gridx,gridy)).T))
    mask = np.array([True if s.contains(point) else False for point in points])
    mask = mask.reshape((y.shape[0], x.shape[0]))

    return mask

def prepare_X_data(X, auto_date_slice, cov_date_slice, 
                   cov_bounds=((90, -90), (-180, 180)),
                   auto_bounds=((90, -90), (-180, 180)),
                   auto_coarsen=None):  # HERE

    Xcov_dm = DataManager(X['cov'], cov_date_slice[0], cov_date_slice[1], cov_bounds, 'cov')

    Xauto_dm = DataManager(X['auto'], auto_date_slice[0], auto_date_slice[1], auto_bounds, 'auto', coarsen=auto_coarsen)  # HERE

    Xdm = {'auto': Xauto_dm, 'cov': Xcov_dm}

    return Xdm

def prepare_Y_data(Y, date_slice, region_bounds=((90, -90), (-180, 180)), coarsen=None):  # HERE

    Ydm = DataManager(Y, date_slice[0], date_slice[1], region_bounds, 'pred', coarsen=coarsen)  # HERE

    return Ydm

def make_dir(dir_name):

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def month2onehot(months):

    one_hot = np.zeros((len(months), 12))
    one_hot[np.arange(len(months)), months-1] = 1

    return one_hot

def compute_anomalies(arr, mn, std, reverse=False):

    if not reverse:
        arr = (arr - mn)/std
    else:
        arr = arr*std + mn

    return arr

def scalar2categ(data, ncategs, matrix_type='one-hot', q33=None, q66=None, count=False):

    """
    data: ndarray of shape [B, lat, lon] or [B, N, lat, lon]
    q33 and q66: float or ndarray of shape [B, lat, lon]
    """

    B = data.shape[0]
    nlat = data.shape[-2]
    nlon = data.shape[-1]
    if len(data.shape) == 3:
        data_flattened = data.reshape((-1, nlat*nlon))
    else:
        nsampl = data.shape[1]
        data_flattened = data.reshape((-1, nsampl, nlat*nlon))

    categs = np.full((B, ncategs, nlat*nlon), np.nan)

    if q33 is None:
        q33_arr = np.full((B, data_flattened.shape[-1]), np.nan)
        q66_arr = np.full((B, data_flattened.shape[-1]), np.nan)
    else:
        if isinstance(q33, float):
            q33_arr = np.full((B, data_flattened.shape[-1]), q33)
            q66_arr = np.full((B, data_flattened.shape[-1]), q66)
        else: 
            q33_arr = q33.reshape((B, data_flattened.shape[-1]))
            q66_arr = q66.reshape((B, data_flattened.shape[-1]))
    
    for p in np.arange(data_flattened.shape[-1]):
        data_point = data_flattened[..., p]
        if not np.all(np.isnan(data_point)):
            if q33 is None and q66 is None:
                q33_ = np.quantile(data_point, 1/3)
                q66_ = np.quantile(data_point, 2/3)
                q33_arr[:, p] = q33_
                q66_arr[:, p] = q66_
            else:
                q33_ = q33_arr[:, p]
                q66_ = q66_arr[:, p]
            if not count:
                categs[:, 0, p] = np.where(data_point<=q33_, 1, 0)
                categs[:, 1, p] = np.where((data_point<=q66_) & (data_point>q33_), 1, 0)
                categs[:, 2, p] = np.where(data_point>q66_, 1, 0)
            else:
                categs[:, 0, p] = np.where(data_point<=q33_[:, None], 1, 0).mean(1)
                categs[:, 1, p] = np.where((data_point<=q66_[:, None]) & (data_point>q33_[:, None]), 1, 0).mean(1)
                categs[:, 2, p] = np.where(data_point>q66_[:, None], 1, 0).mean(1)

    categs = categs.reshape((B, ncategs, nlat, nlon))
    q33_arr = q33_arr.reshape((B, nlat, nlon))
    q66_arr = q66_arr.reshape((B, nlat, nlon))

    if matrix_type == 'sparse':
        nan_mask = np.isnan(categs[0, 0])
        categs = np.argmax(categs, 1)  # index starting from 0
        categs[:, nan_mask] = -999

    return categs, q33_arr, q66_arr

def set_seed(seed: int = 53) -> None:
    """
    Function taken from https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    This article shows the high variability of validation scores when using different seeds.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def printProgressBar(iteration, total, text, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}{text}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


# Classes
class DataManager:
    """
    A class for managing data and performing various operations on it.

    Parameters:
    - X (xr.DataArray): The input data array.
    - idate (str): The start date of the time window.
    - fdate (str): The end date of the time window.
    - boundaries (tuple): The boundaries for latitude and longitude selection.
    - var_type (str): The type of variable.
    - coarsen (int): The coarsening factor for the latitude and longitudes. Default is None.  # HERE

    Attributes:
    - years (np.ndarray): The array of years within the specified time window.
    - boundaries (tuple): The boundaries for latitude and longitude selection.
    - vtype (str): The type of variable.
    - lat (np.ndarray): The latitude values of the selected data.
    - lon (np.ndarray): The longitude values of the selected data.
    - mn (xr.DataArray): The mean values computed based on the specified base period.
    - std (xr.DataArray): The standard deviation values computed based on the specified base period.
    - q33 (xr.DataArray): The 33rd percentile values computed based on the specified base period.
    - q66 (xr.DataArray): The 66th percentile values computed based on the specified base period.
    - min (xr.DataArray): The minimum values computed based on the specified base period.
    - max (xr.DataArray): The maximum values computed based on the specified base period.

    Methods:
    - create_subsamples(names: List[str], samples: List[list], name: str='var'): Creates subsamples based on the specified names and samples.
    - apply_detrend(): Applies detrending to the data.
    - compute_statistics(base_period: Union[list, Dict[int, list]], statistics: List[str], name: str='var'): Computes statistics based on the specified base period and statistics.
    - to_categs(name): Converts the data to categorical values.
    - to_anomalies(name, stdized: bool=False): Converts the data to anomalies.
    - to_range(name, a=-1, b=1): Converts the data to a specified range.
    - monthly2seasonal(name, statistic: str='mean', overlapping: bool=False): Converts the data from monthly to seasonal values.
    - add_season_dim(name): Adds a season dimension to the data.
    - add_seq_dim(name, time_steps, lag_lead='lag'): Adds a sequence dimension to the data.
    """

    def __init__(
            self, 
            X: xr.DataArray, 
            idate: str, 
            fdate: str, 
            boundaries: tuple,
            var_type: str,
            coarsen: int=None):  # HERE

        self['var'] = X
        iyear = int(idate[0:4])
        fyear = int(fdate[0:4])
        self.years = np.arange(iyear, fyear+1)
        self.boundaries = boundaries
        self.vtype = var_type
        self.coarsen = coarsen

        # Transform DataFrame into DataArray
        if isinstance(self['var'], pd.DataFrame):
            self['var'] = self['var'].to_xarray()
            self['var'] = self['var'].to_array("indices", name="all_indices").assign_coords(
                indices=list(self['var'].keys())
            ).transpose('time', 'indices')

        # Selects time window
        self['var'] = self['var'].sel(time=slice(idate, fdate))

        if len(self['var'].shape) > 2:
            # Reverts lat if lat goes from small to high values
            if self['var'].coords['lat'][0] < self['var'].coords['lat'][-1]:
                self['var'] = self['var'].reindex(lat=self['var'].lat[::-1])
            
            self['var'] = self['var'].sel(
                lat=slice(boundaries[0][0], boundaries[0][1]), 
                lon=slice(boundaries[1][0], boundaries[1][1])
            )
            if coarsen is not None:
                self['var'] = self['var'].coarsen(lat=coarsen, lon=coarsen, boundary='exact').mean()

            self.lat = self['var'].lat.values
            self.lon = self['var'].lon.values

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)

    def create_subsamples(self, names: List[str], samples: List[list], name: str='var'):
        """
        Creates subsamples based on the specified names and samples.

        Parameters:
        - names (List[str]): The names of the subsamples.
        - samples (List[list]): The samples to be used for creating the subsamples.
        - name (str): The name of the variable to create subsamples from (default: 'var').
        """
        # for new_name, sample in zip(names, samples):
        #     self[new_name] = deepcopy(self[name][np.isin(self[name]['time.year'], sample)])
        for new_name, sample in zip(names, samples):
            self[new_name] = deepcopy(self[name].sel(time=sample))

    def apply_detrend(self):
        """
        Applies detrending to the data.
        """
        mask_nans = np.isnan(self['var'].values)
        self['var'].values[mask_nans] = 0
        mn = self['var'].values - detrend(self['var'].values, axis=0, type='constant')
        self['var'].values = detrend(self['var'].values, axis=0) + mn
        self['var'].values[mask_nans] = np.nan

    def compute_statistics(self, base_period:Union[list, Dict[int, list]], statistics:List[str], name:str='var'):
        """
        Computes statistics based on the specified base period and statistics.

        If base_period is a dictionary, it computes the statistics sample-wise, i.e, {1951: range(1920, 1950), 1952: range(1921, 1951), ...},
        means that the statistics for the year 1951 are computed on the period between 1920 - 1950, for 1952 they are computed on the
        period between 1920 - 1950, etc.
        If base_period is a list, the statistics are computed on a fixed base period for all samples.

        Parameters:
        - base_period (Union[list, Dict[int, list]]): The base period for computing the statistics.
        - statistics (List[str]): The statistics to be computed.
        - name (str): The name of the variable to compute statistics on (default: 'var').
        """
        if isinstance(base_period, dict):
            if 'mean' in statistics:
                self.mn = xr.concat([self[name][np.isin(self[name]['time.year'], cyrs)].groupby('time.month').mean('time')
                for cyrs in list(base_period.values())], dim=pd.Index(list(base_period.keys()), name='year'))
                self.mn = self.mn.stack({'time': ['year', 'month']}, create_index=False).transpose('time', ...)[0:len(self[name].time)].assign_coords(time=self[name].time)
            if 'std' in statistics:
                self.std = xr.concat([self[name][np.isin(self[name]['time.year'], cyrs)].groupby('time.month').std('time')
                for cyrs in list(base_period.values())], dim=pd.Index(list(base_period.keys()), name='year'))
                self.std = self.std.stack({'time': ['year', 'month']}, create_index=False).transpose('time', ...)[0:len(self[name].time)].assign_coords(time=self[name].time)
            if 'terciles' in statistics:
                terciles = xr.concat([self[name][np.isin(self[name]['time.year'], cyrs)].groupby('time.month').quantile([1/3, 2/3], 'time')
                for cyrs in list(base_period.values())], dim=pd.Index(list(base_period.keys()), name='year'))
                terciles = terciles.stack({'time': ['year', 'month']}, create_index=False).transpose('time', ...)[0:len(self[name].time)].assign_coords(time=self[name].time)
                self.q33 = terciles.isel(quantile=0).drop('quantile').squeeze()
                self.q66 = terciles.isel(quantile=1).drop('quantile').squeeze()
            if 'min' in statistics:
                self.min = xr.concat([self[name][np.isin(self[name]['time.year'], cyrs)].groupby('time.month').min('time')
                for cyrs in list(base_period.values())], dim=pd.Index(list(base_period.keys()), name='year'))
                self.min = self.min.stack({'time': ['year', 'month']}, create_index=False).transpose('time', ...)[0:len(self[name].time)].assign_coords(time=self[name].time)
            if 'max' in statistics:
                self.max = xr.concat([self[name][np.isin(self[name]['time.year'], cyrs)].groupby('time.month').max('time')
                for cyrs in list(base_period.values())], dim=pd.Index(list(base_period.keys()), name='year'))
                self.max = self.max.stack({'time': ['year', 'month']}, create_index=False).transpose('time', ...)[0:len(self[name].time)].assign_coords(time=self[name].time)
        else:
            if 'mean' in statistics:
                self.mn = self[name][np.isin(self[name]['time.year'], base_period)].groupby('time.month').mean('time')
                self.mn = xr.concat([self.mn.sel(month=i) for i in self[name]['time.month'].values], dim=self[name].time).drop('month')
            if 'std' in statistics:
                self.std = self[name][np.isin(self[name]['time.year'], base_period)].groupby('time.month').std('time')
                self.std = xr.concat([self.std.sel(month=i) for i in self[name]['time.month'].values], dim=self[name].time).drop('month')
            if 'terciles' in statistics:
                terciles = self[name][np.isin(self[name]['time.year'], base_period)].groupby('time.month').quantile([1/3, 2/3])
                terciles = xr.concat([terciles.sel(month=i) for i in self[name]['time.month'].values], dim=self[name].time).drop('month')
                self.q33 = terciles.isel(quantile=0).drop('quantile').squeeze()
                self.q66 = terciles.isel(quantile=1).drop('quantile').squeeze()
            if 'min' in statistics:
                self.min = self[name][np.isin(self[name]['time.year'], base_period)].groupby('time.month').min('time')
                self.min = xr.concat([self.min.sel(month=i) for i in self[name]['time.month'].values], dim=self[name].time).drop('month')
            if 'max' in statistics:
                self.max = self[name][np.isin(self[name]['time.year'], base_period)].groupby('time.month').max('time')
                self.max = xr.concat([self.max.sel(month=i) for i in self[name]['time.month'].values], dim=self[name].time).drop('month')

    def to_categs(self, name):
        """
        Converts the data to categorical values.

        Parameters:
        - name (str): The name of the variable to convert to categorical values.
        """
        mask = np.isnan(self[name][0, :, :])
        one_hot = np.full((self[name].shape[0], 3, self[name].shape[1], self[name].shape[2]), np.nan)
        one_hot[:, 0, :, :] = np.where(self[name]<=self.q33, 1, 0)
        one_hot[:, 1, :, :] = np.where((self[name]<=self.q66)&(self[name]>self.q33), 1, 0)
        one_hot[:, 2, :, :] = np.where(self[name]>self.q66, 1, 0)
        self[name][:] = np.argmax(one_hot, 1)
        self[name].values[..., mask] = np.nan
            
    def to_anomalies(self, name, stdized: bool=False, reverse_operation: bool=False):
        """
        Converts the data to anomalies.

        Parameters:
        - name (str): The name of the variable to convert to anomalies.
        - stdized (bool): Whether to standardize the anomalies (default: False).
        - reverse_operation (bool): Whether to reverse the operation (default: False).
        """
        if not reverse_operation:
            if stdized:
                self[name][:] = ((self[name] - self.mn.sel(time=self[name].time))/self.std.sel(time=self[name].time)).values
            else:
                self[name][:] = (self[name] - self.mn.sel(time=self[name].time)).values
        else:
            if stdized:
                self[name][:] = (self[name]*self.std.sel(time=self[name].time) + self.mn.sel(time=self[name].time)).values
            else:
                self[name][:] = (self[name] + self.mn.sel(time=self[name].time)).values
    
    def to_range(self, name, a=-1, b=1, reverse_operation=False):
        """
        Converts the data to a specified range.

        Parameters:
        - name (str): The name of the variable to convert to a specified range.
        - a (float): The lower bound of the range (default: -1).
        - b (float): The upper bound of the range (default: 1).
        - reverse_operation (bool): Whether to reverse the operation (default: False).
        """
        if not reverse_operation:
            self[name][:] = (((b-a)*(self[name] - self.min.sel(time=self[name].time))/(self.max.sel(time=self[name].time) - self.min.sel(time=self[name].time))) + a).values
        else:
            self[name][:] = (((self[name] - a)/(b - a))*(self.max.sel(time=self[name].time) - self.min.sel(time=self[name].time)) + self.min.sel(time=self[name].time)).values
            
    def monthly2seasonal(self, name, statistic: str='mean', overlapping: bool=False):
        """
        Converts the data from monthly to seasonal values.

        Parameters:
        - name (str): The name of the variable to convert from monthly to seasonal values.
        - statistic (str): The statistic to be computed for the seasonal values (default: 'mean').
        - overlapping (bool): Whether the seasonal values should overlap (default: False).
        """
        if statistic == 'mean':
            self[name] = self[name].rolling(time=3, center=True).mean('time')[1:-1]
        elif statistic == 'sum':
            self[name] = self[name].rolling(time=3, center=True).sum('time')[1:-1]
        if not overlapping:
            self[name] = self[name][1::3]
    
    def add_season_dim(self, name):
        """
        Adds a season dimension to the data.

        Parameters:
        - name (str): The name of the variable to add a season dimension to.
        """
        int2seas = {1: 'DJF', 2: 'JFM', 3:'FMA', 4: 'MAM', 5: 'AMJ', 6: 'MJJ', 
                    7: 'JJA', 8: 'JAS', 9: 'ASO', 10: 'SON', 11: 'OND', 12: 'NDJ'}
        seas = [int2seas[i] for i in np.unique(self[name].time.dt.month.values)]
        self[name] = xr.concat([i[1].assign_coords(time=seas)
                                for i in list(self[name].groupby('time.year'))], 
                                dim=pd.Index(np.unique(self[name].time.dt.year.values), name='samples'))
    
    def add_seq_dim(self, name, time_steps, lag_lead='lag'):
        """
        Adds a sequence dimension to a given variable in the dataset.

        Parameters:
            name (str): The name of the variable to add the sequence dimension to.
            time_steps (int): The number of time steps to include in each sequence.
            lag_lead (str): The type of sequence to create (default: 'lag').
        """

        if lag_lead == 'lag':
            seq_dim = range(-time_steps+1, 1)
            self[name] = xr.concat(
            [self[name].fillna(-999.).shift(time=i) for i in range(time_steps)[::-1]], 
             dim=pd.Index(seq_dim, name='time_seq')).transpose('time', ...).dropna(dim='time', how='any')
        elif lag_lead == 'lead':
            seq_dim = range(time_steps)
            seqs = [self[name].fillna(-999.).shift(time=-i) for i in range(time_steps)]
            self[name] = xr.concat(
            seqs, 
             dim=pd.Index(seq_dim, name='time_seq')).transpose('time', ...).dropna(dim='time', how='any')
        self[name] = xr.where(self[name]==-999., np.nan, self[name])

    def replace_nan(self, name, replace_value):
        """
        Replaces NaN values in the data.

        Parameters:
            name (str): The name of the variable to replace NaN values in.
            replace_value: The value to replace NaN values with.
        """

        self.spatial_mask = np.isnan(self[name][0, :, :])
        self[name].values[..., self.spatial_mask] = replace_value
   
    def spatio_flatten(self, name, drop_nan=False):
        """
        Flattens the specified variable along the spatial dimensions.

        Parameters:
        - name (str): The name of the variable to flatten.
        - drop_nan (bool): Whether to drop NaN values after flattening. Default is False.
        """

        self[name] = self[name].stack(point=('lat', 'lon'))
        if drop_nan:
            self[name] = self[name].dropna(dim='point')


class CreateDataset(Dataset):

    def __init__(self, X, Y, DEVICE):
        self.DEVICE = DEVICE
        self.X = [torch.as_tensor(x.astype(np.float32), device=self.DEVICE)
                  for x in X]
        self.Y = torch.as_tensor(Y.astype(np.float32), device=self.DEVICE)

    def __len__(self):

        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        
        X = [x[idx] for x in self.X]
        Xmask = torch.where(X[0][0:1]==-999., 0., 1.).to(self.DEVICE)
        Y = self.Y[idx]

        return X, Xmask, Y
    
class EvalMetrics:

    def RPS(ytrue, ypred, ax=0):

        RPS = np.nanmean(((np.cumsum(ypred, 0) - np.cumsum(ytrue, 0))**2).sum(0), axis=ax)
        
        return RPS

    def Bias(y_true, y_pred, ax=0):

        return np.nanmean((y_true.squeeze() - y_pred.squeeze()), axis=ax)


    def MSE(y_true, y_pred, ax=0):

        return np.nanmean((y_true.squeeze() - y_pred.squeeze())**2, axis=ax)

    def RMSE(y_true, y_pred, ax=0):

        return np.sqrt(np.nanmean((y_true.squeeze() - y_pred.squeeze())**2, axis=ax))

    def RMEV(y_true, y_pred, ax=0):
        "root mean ensemble variance"

        return np.sqrt(np.nanmean(np.nansum((y_true[:, None] - y_pred)**2, 1)/(y_pred.shape[1]-1), axis=ax))
    
    def SSR(y_true, y_pred, ax=0):

        return np.sqrt((y_pred.shape[0]+1)/y_pred.shape[0])*EvalMetrics.RMEV(y_true, y_pred, ax)/EvalMetrics.RMSE(y_true, y_pred.mean(1), ax)
    
    def spread_skill_ratio(y_true, y_pred, ax=0):

        return np.sqrt((y_pred.shape[0]+1)/y_pred.shape[0])*EvalMetrics.RMEV(y_true, y_pred, ax)/EvalMetrics.RMSE(y_true, y_pred.mean(1), ax)

    def obs_rank(y_true, y_pred, extreme_threshold=None):
        """
        observation rank for rank histogram
        y_true: ndarray of shape [nobs, nlat, nlon]
        y_pred: ndarray of shape [nobs, nmembs, nlat, nlon]
        """

        nobs = y_pred.shape[0]
        nmembs = y_pred.shape[1]
        mask = ~np.isnan(y_true[0].reshape(-1))

        data_c = np.concatenate((y_pred, y_true[:, np.newaxis]), 1)
        data_c = data_c.reshape(nobs, nmembs+1, -1)[:, :, mask]  # selects only valid grid points
        data_c = data_c.transpose((1, 0, 2)).reshape((nmembs+1, -1))  # (nmembs+1, nobs*nvalid)
        if extreme_threshold is not None:
            ev_mask = np.where(np.abs(data_c[-1]) >= extreme_threshold, True, False)
        else:
            ev_mask = np.ones(data_c.shape[1], dtype=bool)
        ranks = np.argsort(data_c, 0)
        obsrank = np.array([np.nonzero(ranks[:, j] == nmembs)[0][0] for j in range(ranks.shape[1])])
        # obsrank = np.nonzero((ranks == nmembs))
        ranknorm = (obsrank+1)/(nmembs+1)
        ranknorm[~ev_mask] = np.nan

        return ranknorm
    
    def calibration_refinement_functions(y_true, y_pred, bins):
        """
        compute calibration and refinement functions for 
        reliability and sharpness diagrams
        y_true: ndarray of shape [ncategs, nobs, nlat, nlon]
        y_pred: ndarray of shape [ncategs, nobs, nlat, nlon]
        """

        y_true = y_true.reshape(y_true.shape[0], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        mask = ~np.isnan(y_true[0])

        ncategs = y_pred.shape[0]
        nbins = len(bins)-1

        pred_marginal = np.full((ncategs, nbins), np.nan)
        obs_freqs = np.full((ncategs, nbins), np.nan)
        avg_probs = np.full((ncategs, nbins), np.nan)
        rel = np.full((ncategs), np.nan)
        res = np.full((ncategs), np.nan)

        for i in range(ncategs):
            yt = y_true[i][mask]
            yp = y_pred[i][mask]
            n = []
            for j in range(1, nbins+1):
                if j < nbins-1:
                    idx = np.argwhere((yp >= bins[j-1]) & (yp < bins[j]))
                else:
                    idx = np.argwhere((yp >= bins[j-1]) & (yp <= bins[j]))
                yp_ = yp[idx]
                yt_ = yt[idx]
                pred_marginal[i, j-1] = len(yp_)/len(yp)
                obs_freqs[i, j-1] = np.mean(yt_)
                avg_probs[i, j-1] = np.mean(yp_)
                n.append(len(idx))

            n = np.array(n)
            rel[i] = np.sum(n*((avg_probs[i, :] - obs_freqs[i, :])**2))/np.sum(n)
            res[i] = np.sum(n*((obs_freqs[i, :] - (np.sum(n*obs_freqs[i, :])/np.sum(n)))**2))/np.sum(n)
        
        return obs_freqs, avg_probs, pred_marginal, rel, res
