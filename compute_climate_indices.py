import argparse
from copy import deepcopy
from datetime import datetime
import os
import pandas as pd
import xarray as xr
import numpy as np
from modules.rotator import Rotator
from sklearn.decomposition import PCA
from scipy.signal import detrend

"""
Oceanic indices: ONI, ATN, ATS, ATL, IOBW, IOB, PDO, SASDI
Atmospheric indices: AO, AAO, NAO, PNA, PSA1, PSA2, 
"""

telnet_datadir = os.getenv('TELNET_DATADIR')
era5_dir = os.path.join(telnet_datadir, 'era5')  # Legacy, not used with ARCO ERA5


class LinearRegression:

    def __init__(self, X, y, theta=[]):

        self.X = X
        self.y = y
        self.n = y.shape[0]
        if len(theta) == 0:
            self.theta = np.zeros((X.shape[1]))
        else:
            self.theta = theta
        self.H = self.X.T@self.X
        self.H_inv = np.linalg.inv(self.H)

    def predict(self, X=[], verbose=0):

        if len(X) == 0:
            return self.X@self.theta
        else:
            return X@self.theta

    def unconstrained_fit(self):

        self.theta = self.H_inv@self.X.T@self.y
    
    def residual_variance(self):

        return np.sum((self.y - self.predict())**2)/(self.n-2)

    def constrained_fit(self, A, b):

        """
        Solve least square fit subject to linear constraint in the form of A@theta = b
        """

        self.unconstrained_fit()
        
        t = self.H_inv@A.T@np.linalg.inv(A@self.H_inv@A.T)

        return self.theta + b*t - t@A@self.theta

class MCA:

    def __init__(self, X, Y):
        
        self.X = X
        self.Y = Y
        self.N = self.X.shape[0]
        self.I = self.X.shape[1]
        self.J = self.Y.shape[1]
        self.M = min(self.I, self.J)
    
    def joint_matrix(self):

        self.Z = np.concatenate((self.X, self.Y), 1)

    def joint_cov(self):

        N = self.N
        I = self.I

        S = (1/(N-1)) * (self.Z.T @ self.Z)
        self.S = S
        self.Sxx = S[:I, :I]
        self.Sxy = S[:I, I:]
        self.Syx = S[I:, :I]
        self.Syy = S[I:, I:]

    def MCA_vectors(self):
        
        M = self.M
        Sxy = self.Sxy

        LSV, max_cov, RSV_T = np.linalg.svd(Sxy)   # Left singular vectors (LSV), maxiximized covariance (max_cov) and right singular vectors transposed (RSV_T)

        self.L = (LSV[:, :M])
        self.R = (RSV_T.T[:, :M])
        self.cov = np.diag(max_cov[:M])

    def MCA_variables(self, Xn=[], Yn=[]):

        if len(Xn) != 0:
            V = Xn@self.L
            W = Yn@self.R
            return V, W
        else:
            self.V = self.X@self.L
            self.W = self.Y@self.R

def detrend_darray(darr):

    mask_nans = np.isnan(darr.values)
    darr.values[mask_nans] = 0
    mn = darr.values - detrend(darr.values, axis=0, type='constant')
    darr.values = detrend(darr.values, axis=0) + mn
    darr.values[mask_nans] = np.nan

    return darr

def compute_joint_mask(data1, data2):

    mask = np.logical_and(~np.isnan(data1).reshape(data1.shape[0], -1).any(axis=0), ~np.isnan(data2).reshape(data2.shape[0], -1).any(0))

    return mask

def compute_rpca(data, n_components=10):
    """
    Compute the RPCA of the data.
    
    Parameters:
    -----------
    data: xarray.DataArray
        The input data.
    n_components: int
        The number of components to keep.
    
    Returns:
    --------
    xarray.DataArray
        The RPCA of the data.
    """

    # Compute the RPCA
    rpca = PCA(n_components=n_components)
    rpca.fit(data)
    # Rotate loadings with varimax
    rotator = Rotator(method='varimax')
    rotated_loadings = rotator.fit_transform(rpca.components_.T)
    
    return rotated_loadings.T

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    from numpy import eye, asarray, dot, sum
    from scipy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, np.diag(np.diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return dot(Phi, R), R

def orthonormal_rotation(L, R, cov, Ni):

        """
        Orthogonal Rotation of Spatial Patterns Derived from Singular Value Decomposition Analysis. Cheng and Dunkerton (1995)
        """
        
        Ncov_sqrt = np.sqrt(cov[0:Ni, 0:Ni])
        P = np.concatenate([L, R], 0)[:, 0:Ni]@Ncov_sqrt
        P_, U = varimax(P)
        Lrot_ = L[:, 0:Ni]@Ncov_sqrt@U
        Rrot_ = R[:, 0:Ni]@Ncov_sqrt@U
        Wl = np.diag(np.diag(np.sqrt(Lrot_.T@Lrot_)))
        Wr = np.diag(np.diag(np.sqrt(Rrot_.T@Rrot_)))
        Lrot = Lrot_@np.linalg.inv(Wl)
        Rrot = Rrot_@np.linalg.inv(Wr)
        covrot = Wl@Wr

        return Lrot, Rrot, covrot

def compute_pca(data, n_components=10):
    """
    Compute the PCA of the data.

    """

    # Compute the PCA
    pca = PCA(n_components=n_components)
    pca.fit(data)
    
    return pca.components_

def transform_data(data, loadings):

    transformed_data = np.dot(data.reshape(data.shape[0], -1), loadings.reshape(-1, 1))

    return transformed_data

def compute_oni(ssta):
    """
    Compute the Oceanic Nino Index (ONI) from the data.

    """

    ssta.coords['lon'] = (ssta.coords['lon'] + 180) % 360 - 180
    ssta = ssta.sortby(ssta.lon)

    # Compute the ONI
    oni = ssta.sel(lat=slice(5.5, -5.5), lon=slice(-170, -120)).mean(dim=['lat', 'lon'], skipna=True)
    
    return oni.squeeze()

def compute_std_anoms(darray, base_period, stded=False):

    mn = darray.sel(time=base_period).groupby("time.month").mean("time")
    if stded:
        std = darray.sel(time=base_period).groupby("time.month").std("time")
    else:
        std = 1
    
    X = xr.apply_ufunc(
    lambda x, m, s: (x - m) / s,
    darray.groupby("time.month"),
    mn,
    std,
    )

    return X

def remove_oni_signal(sst, u10, v10, oni, land_mask=None):

    sst_r = sst.reshape((sst.shape[0], sst.shape[1]*sst.shape[2]))
    if land_mask is None:
        land_mask = ~np.isnan(sst_r).any(axis=0)
    sst_r = sst_r[:, land_mask]
    u10_r = u10.reshape((u10.shape[0], u10.shape[1]*u10.shape[2]))
    u10_r = u10_r[:, land_mask]
    v10_r = v10.reshape((v10.shape[0], v10.shape[1]*v10.shape[2]))
    v10_r = v10_r[:, land_mask]

    # Removing linear signal of ONI from fields
    for i in np.arange(sst_r.shape[1]):
        sstfit = LinearRegression(oni, sst_r[:, i])
        sstfit.unconstrained_fit()
        sst_t = sstfit.theta
        sst_r[:, i] = sst_r[:, i] - oni@sst_t

        u10fit = LinearRegression(oni, u10_r[:, i])
        u10fit.unconstrained_fit()
        u10_t = u10fit.theta
        u10_r[:, i] = u10_r[:, i] - oni@u10_t
        
        v10fit = LinearRegression(oni, v10_r[:, i])
        v10fit.unconstrained_fit()
        v10_t = v10fit.theta
        v10_r[:, i] = v10_r[:, i] - oni@v10_t
    
    return sst_r, u10_r, v10_r, land_mask

def preprocess_data(data, lat_weights, base_period, land_mask=None):
    
    data = data.sel(time=base_period)
    data = data.values.reshape(data.shape[0], -1)*lat_weights
    if land_mask is None:
        land_mask = ~np.isnan(data).any(axis=0)

    return data[:, land_mask], land_mask

def compute_tropical_atlantic_indices(ssta, u10a, v10a, oni, ersst_ssta, e5_u10a, e5_v10a, base_period):

    oni_ersst = compute_oni(ersst_ssta)

    ssta.coords['lon'] = (ssta.coords['lon'] + 180) % 360 - 180
    ssta = ssta.sortby(ssta.lon)
    u10a.coords['lon'] = (u10a.coords['lon'] + 180) % 360 - 180
    u10a = u10a.sortby(u10a.lon)
    v10a.coords['lon'] = (v10a.coords['lon'] + 180) % 360 - 180
    v10a = v10a.sortby(v10a.lon)

    ssta_atl = ssta.sel(lat=slice(32, -22), lon=slice(-74, 20))
    lats = ssta_atl.lat.values
    lat_weights = np.tile(np.sqrt(np.cos(np.deg2rad(lats)))[:, np.newaxis], (1, ssta_atl.lon.shape[0])).reshape(-1)
    lons = ssta_atl.lon.values
    time = ssta_atl.time
    u10a_atl = u10a.sel(lat=lats, lon=lons)
    v10a_atl = v10a.sel(lat=lats, lon=lons)

    ersst_ssta.coords['lon'] = (ersst_ssta.coords['lon'] + 180) % 360 - 180
    ersst_ssta = ersst_ssta.sortby(ersst_ssta.lon)
    e5_u10a.coords['lon'] = (e5_u10a.coords['lon'] + 180) % 360 - 180
    e5_u10a = e5_u10a.sortby(e5_u10a.lon)
    e5_v10a.coords['lon'] = (e5_v10a.coords['lon'] + 180) % 360 - 180
    e5_v10a = e5_v10a.sortby(e5_v10a.lon)
    ersst_ssta_atl = ersst_ssta.sel(lat=lats, lon=lons)
    e5_u10a_atl = e5_u10a.sel(lat=lats, lon=lons)
    e5_v10a_atl = e5_v10a.sel(lat=lats, lon=lons)
    
    land_mask = compute_joint_mask(ssta_atl.values, ersst_ssta_atl.values)

    # Eigenvector computation based on ERSST and ERA5 data
    
    sst_r, u10_r, v10_r, _ = remove_oni_signal(ersst_ssta_atl.sel(time=slice(base_period[0], base_period[1])).values, 
                                                      e5_u10a_atl.sel(time=slice(base_period[0], base_period[1])).values, 
                                                      e5_v10a_atl.sel(time=slice(base_period[0], base_period[1])).values, 
                                                      oni_ersst.sel(time=slice(base_period[0], base_period[1])).values[:, np.newaxis],
                                                      land_mask)
    X = sst_r*lat_weights[land_mask]
    Y = np.append(u10_r*lat_weights[land_mask], v10_r*lat_weights[land_mask], 1)

    n_components = 4
    MCA_obj = MCA(X, Y)  # TODO: Include MCA class
    MCA_obj.joint_matrix()
    MCA_obj.joint_cov()
    MCA_obj.MCA_vectors()

    L = deepcopy(MCA_obj.L)
    R = deepcopy(MCA_obj.R)
    cov = deepcopy(MCA_obj.cov)

    L, R, cov = orthonormal_rotation(L, R, cov, n_components)

    L_2d = np.full((n_components, lats.shape[0]*lons.shape[0]), np.nan)
    R_2d = np.full((n_components, 2*lats.shape[0]*lons.shape[0]), np.nan)
    for p in range(n_components):
        L_2d[p, land_mask] = L[:, p]
        R_2d[p, np.tile(land_mask, (2))] = R[:, p]

    # Computing PCs based on model's data project onto the MCA vectors
    sst_r, u10_r, v10_r, _ = remove_oni_signal(ssta_atl.values, 
                                                      u10a_atl.values, 
                                                      v10a_atl.values, 
                                                      oni.values[:, np.newaxis],
                                                      land_mask)
    X = sst_r*lat_weights[land_mask]
    W1 = X@(L_2d[0][land_mask])
    W2 = X@(L_2d[2][land_mask])

    ATS = W1/W1.std()
    ATN = W2/W2.std()

    return ATN.squeeze(), ATS.squeeze()

def compute_indian_ocean_indices(ssta, ersst_ssta, oni, base_period):

    indian_lat = slice(20.5, -20.5)
    indian_lon = slice(40.5, 120.5)

    ssta_indian = ssta.sel(lat=indian_lat, lon=indian_lon)
    ersst_ssta_indian = ersst_ssta.sel(lat=indian_lat, lon=indian_lon)

    ssta_indian.coords['lon'] = (ssta_indian.coords['lon'] + 180) % 360 - 180
    ssta_indian = ssta_indian.sortby(ssta_indian.lon)

    ersst_ssta_indian.coords['lon'] = (ersst_ssta_indian.coords['lon'] + 180) % 360 - 180
    ersst_ssta_indian = ersst_ssta_indian.sortby(ersst_ssta_indian.lon)

    land_mask = compute_joint_mask(ssta_indian.values, ersst_ssta_indian.values)
 
    lats = ssta_indian.lat.values
    lat_weights = np.tile(np.sqrt(np.cos(np.deg2rad(lats)))[:, np.newaxis], (1, ssta_indian.lon.shape[0])).reshape(-1)
    lons = ssta_indian.lon.values
    time = ssta_indian.time

    n_components = 2
    X, _ = preprocess_data(ersst_ssta_indian, lat_weights, slice(base_period[0], base_period[1]), land_mask)
    loading = compute_pca(X, n_components=n_components)

    # Compute the IOBW and IOD,
    X, _ = preprocess_data(ssta_indian, lat_weights, slice(None, None), land_mask)
    IOBW = transform_data(X, loading[0])
    IOD = transform_data(X, loading[1]*-1)

    IOBW = IOBW/IOBW.std()
    IOD = IOD/IOD.std()

    return IOBW.squeeze(), IOD.squeeze()

def compute_sasdi(ssta):

    ssta.coords['lon'] = (ssta.coords['lon'] + 180) % 360 - 180
    ssta = ssta.sortby(ssta.lon)
    
    ssta_atl = ssta.sel(lat=slice(-15.5, -40.5), lon=slice(-30.5, 0.5))
    lats = ssta_atl.lat.values
    lat_weights = np.sqrt(np.cos(np.deg2rad(np.tile(lats[:, np.newaxis], (1, ssta_atl.lon.shape[0])))))
    lons = ssta_atl.lon.values
    time = ssta_atl.time

    ssta_atl = ssta_atl*lat_weights

    SW = ssta.sel(lat=slice(-30.5, -40.5), lon=slice(-30.5, -10.5)).mean(['lat', 'lon'], skipna=True)
    NE = ssta.sel(lat=slice(-15.5, -25.5,), lon=slice(-20.5, 0.5)).mean(['lat', 'lon'], skipna=True)

    SASDI = SW - NE

    SASDI = SASDI/SASDI.std()

    return SASDI.squeeze()

def compute_pdo(ssta, ersst_ssta, base_period):

    lat_pcf = slice(55.5, 20.5)
    lon_pcf = slice(45, 130)

    ssta_pcf = ssta.sel(lat=lat_pcf).isel(lon=lon_pcf)
    lats = ssta_pcf.lat.values
    lat_weights = np.tile(np.sqrt(np.cos(np.deg2rad(lats)))[:, np.newaxis], (1, ssta_pcf.lon.shape[0])).reshape(-1)

    ersst_ssta_pcf = ersst_ssta.sel(lat=lat_pcf).isel(lon=lon_pcf)

    land_mask = compute_joint_mask(ssta_pcf.values, ersst_ssta_pcf.values)

    n_components = 1
    X, _ = preprocess_data(ersst_ssta_pcf, lat_weights, slice(base_period[0], base_period[1]), land_mask)
    loading = compute_pca(X, n_components=n_components)

    # Compute the PDO,
    X, _ = preprocess_data(ssta_pcf, lat_weights, slice(None, None), land_mask)
    PDO = transform_data(X, loading[0])

    PDO = PDO/PDO.std()

    return PDO.squeeze()

def compute_ao_aao(hgt, e5_hgt, base_period):

    hgt.coords['lon'] = (hgt.coords['lon'] + 180) % 360 - 180
    hgt = hgt.sortby(hgt.lon)
    time = hgt.time

    hgt_NH= hgt.sel(level=e5_hgt.level.values[-1], lat=slice(89.5, 19.5))
    lats_NH= hgt_NH.lat.values
    lat_weights_NH= np.tile(np.sqrt(np.cos(np.deg2rad(lats_NH)))[:, np.newaxis], (1, hgt_NH.lon.shape[0])).reshape(-1)
    lons_NH= hgt_NH.lon.values

    hgt_SH = hgt.sel(level=e5_hgt.level.values[1], lat=slice(-19.5, -89.5))
    lats_SH = hgt_SH.lat.values
    lat_weights_SH = np.tile(np.sqrt(np.cos(np.deg2rad(lats_SH)))[:, np.newaxis], (1, hgt_SH.lon.shape[0])).reshape(-1)
    lons_SH = hgt_SH.lon.values

    e5_hgt.coords['lon'] = (e5_hgt.coords['lon'] + 180) % 360 - 180
    e5_hgt = e5_hgt.sortby(e5_hgt.lon)
    e5_hgt_NH = e5_hgt.sel(level=e5_hgt.level.values[-1], lat=slice(89.5, 19.5))
    e5_hgt_SH = e5_hgt.sel(level=e5_hgt.level.values[1], lat=slice(-19.5, -89.5))

    land_mask_NH = compute_joint_mask(hgt_NH.values, e5_hgt_NH.values)
    land_mask_SH = compute_joint_mask(hgt_SH.values, hgt_SH.values)

    X_NH, _ = preprocess_data(e5_hgt_NH, lat_weights_NH, slice(base_period[0], base_period[1]), land_mask_NH)
    X_SH, _ = preprocess_data(e5_hgt_SH, lat_weights_SH, slice(base_period[0], base_period[1]), land_mask_SH)

    # Compute the PCA
    n_components = 1
    ao_loading = compute_pca(X_NH, n_components=n_components)
    aao_loading = compute_pca(X_SH, n_components=n_components)

    ao_loading = ao_loading
    aao_loading = aao_loading

    # Compute indices
    X_NH, _ = preprocess_data(hgt_NH, lat_weights_NH, slice(None, None), land_mask_NH)
    X_SH, _ = preprocess_data(hgt_SH, lat_weights_SH, slice(None, None), land_mask_SH)
    AO = transform_data(X_NH, ao_loading[0])
    AAO = transform_data(X_SH, aao_loading[0])

    AO = AO/AO.std()
    AAO = AAO/AAO.std()

    return AO.squeeze(), AAO.squeeze()

def compute_psa_indices(hgt, e5_hgt, base_period):

    hgt.coords['lon'] = (hgt.coords['lon'] + 180) % 360 - 180
    hgt = hgt.sortby(hgt.lon)
    time = hgt.time

    hgt_south = hgt.sel(level=e5_hgt.level.values[0], lat=slice(-0.5, -89.5))
    lats_SH = hgt_south.lat.values
    lat_weights_SH = np.tile(np.sqrt(np.cos(np.deg2rad(lats_SH)))[:, np.newaxis], (1, hgt_south.lon.shape[0])).reshape(-1)
    lons_SH = hgt_south.lon.values

    e5_hgt.coords['lon'] = (e5_hgt.coords['lon'] + 180) % 360 - 180
    e5_hgt = e5_hgt.sortby(e5_hgt.lon)
    e5_hgt_south = e5_hgt.sel(level=e5_hgt.level.values[0], lat=slice(-0.5, -89.5))

    land_mask_SH = compute_joint_mask(hgt_south.values, e5_hgt_south.values)

    X_SH, _ = preprocess_data(e5_hgt_south, lat_weights_SH, slice(base_period[0], base_period[1]), land_mask_SH)

    # Compute the PCA
    n_components = 3
    loadings = compute_pca(X_SH, n_components=n_components)

    # Compute indices
    X_SH, _ = preprocess_data(hgt_south, lat_weights_SH, slice(None, None), land_mask_SH)
    PSA1 = transform_data(X_SH, loadings[1])
    PSA2 = transform_data(X_SH, loadings[2])

    PSA1 = PSA1/PSA1.std()
    PSA2 = PSA2/PSA2.std()

    return PSA1.squeeze(), PSA2.squeeze()

def compute_nh_indices(hgt, e5_hgt, base_period):

    hgt.coords['lon'] = (hgt.coords['lon'] + 180) % 360 - 180
    hgt = hgt.sortby(hgt.lon)
    time = hgt.time

    hgt_north = hgt.sel(level=e5_hgt.level.values[0], lat=slice(89.5, 19.5))
    lats_NH = hgt_north.lat.values
    lat_weights_NH = np.tile(np.sqrt(np.cos(np.deg2rad(lats_NH)))[:, np.newaxis], (1, hgt_north.lon.shape[0])).reshape(-1)
    lons_NH = hgt_north.lon.values

    e5_hgt.coords['lon'] = (e5_hgt.coords['lon'] + 180) % 360 - 180
    e5_hgt = e5_hgt.sortby(e5_hgt.lon)
    e5_hgt_north = e5_hgt.sel(level=e5_hgt.level.values[0], lat=slice(89.5, 19.5))

    land_mask_NH = compute_joint_mask(hgt_north.values, e5_hgt_north.values)

    X_NH, _ = preprocess_data(e5_hgt_north, lat_weights_NH, slice(base_period[0], base_period[1]), land_mask_NH)

    n_components = 10
    rotated_loadings = compute_rpca(X_NH, n_components=n_components)

    # NAO = loadings 0 and PNA = loadings 1
    X_NH, _ = preprocess_data(hgt_north, lat_weights_NH, slice(None, None), land_mask_NH)
    NAO = transform_data(X_NH, rotated_loadings[0])
    PNA = transform_data(X_NH, rotated_loadings[1])

    NAO = NAO/NAO.std()
    PNA = PNA/PNA.std()

    return NAO.squeeze(), PNA.squeeze()

def compute_climate_indices(final_season):

    # Load the data
    base_period = ('1982-01-01', '2015-12-01')

    # Load ERSSTv5 (single file covering 1940-present)
    ersst_file = os.path.join(telnet_datadir, 'ersstv5_1940-present.nc')
    sstds_ersst = xr.open_dataset(ersst_file).load()
    sst_ersst = sstds_ersst.sst
    sst_ersst = compute_std_anoms(sst_ersst, slice(base_period[0], base_period[1]), stded=False)
    sst_ersst = detrend_darray(sst_ersst)
    sst_ersst = sst_ersst.rolling(time=3, center=True).mean().sel(time=slice('1941-01-01', final_season))

    # Load ERA5 u10, v10, height from ARCO ERA5 (streams from GCS)
    from load_arco_era5 import open_arco_era5
    print("  Loading ERA5 u10 from ARCO ERA5...")
    u10ds_e5 = open_arco_era5('u10', period=('1940-01-01', str(final_season)[:10]))
    # Resample hourly to monthly and load into memory (ARCO returns dask arrays)
    u10_e5 = u10ds_e5.u10.resample(time='MS').mean().compute()
    # Rename coordinates if needed
    if 'latitude' in u10_e5.dims:
        u10_e5 = u10_e5.rename({'latitude': 'lat', 'longitude': 'lon'})
    u10_e5 = compute_std_anoms(u10_e5, slice(base_period[0], base_period[1]), stded=False)
    u10_e5 = detrend_darray(u10_e5)
    u10_e5 = u10_e5.rolling(time=3, center=True).mean().sel(time=slice('1941-01-01', final_season))

    print("  Loading ERA5 v10 from ARCO ERA5...")
    v10ds_e5 = open_arco_era5('v10', period=('1940-01-01', str(final_season)[:10]))
    v10_e5 = v10ds_e5.v10.resample(time='MS').mean().compute()
    if 'latitude' in v10_e5.dims:
        v10_e5 = v10_e5.rename({'latitude': 'lat', 'longitude': 'lon'})
    v10_e5 = compute_std_anoms(v10_e5, slice(base_period[0], base_period[1]), stded=False)
    v10_e5 = detrend_darray(v10_e5)
    v10_e5 = v10_e5.rolling(time=3, center=True).mean().sel(time=slice('1941-01-01', final_season))

    print("  Loading ERA5 geopotential height from ARCO ERA5...")
    hds_e5 = open_arco_era5('height', period=('1940-01-01', str(final_season)[:10]))
    h_e5 = hds_e5.height.resample(time='MS').mean().compute()
    if 'latitude' in h_e5.dims:
        h_e5 = h_e5.rename({'latitude': 'lat', 'longitude': 'lon'})
    h_e5 = compute_std_anoms(h_e5, slice(base_period[0], base_period[1]), stded=False)
    h_e5 = detrend_darray(h_e5)
    h_e5 = h_e5.rolling(time=3, center=True).mean().sel(time=slice('1941-01-01', final_season))

    # Compute the Pacific indices (ONI)
    oni = compute_oni(deepcopy(sst_ersst))
    pdo = compute_pdo(deepcopy(sst_ersst), deepcopy(sst_ersst), base_period)  # Computed on the full data availability before 1981

    # Compute the Atlantic indices
    atn, ats = compute_tropical_atlantic_indices(deepcopy(sst_ersst), deepcopy(u10_e5), deepcopy(v10_e5), oni, deepcopy(sst_ersst), deepcopy(u10_e5), deepcopy(v10_e5), base_period)
    atl = atn - ats
    sasdi = compute_sasdi(deepcopy(sst_ersst))

    # Compute the Indian Ocean indices
    iobw, iod = compute_indian_ocean_indices(deepcopy(sst_ersst), deepcopy(sst_ersst), oni, base_period)

    # # Compute the AO and AAO indices
    ao, aao = compute_ao_aao(deepcopy(h_e5), deepcopy(h_e5), base_period)

    # # Compute PSA indices
    psa1, psa2 = compute_psa_indices(deepcopy(h_e5), deepcopy(h_e5), base_period)

    # # Compute NH indices
    nao, pna = compute_nh_indices(deepcopy(h_e5), deepcopy(h_e5), base_period)

    return oni, pdo, atn, ats, atl, sasdi, iobw, iod, ao, aao, psa1, psa2, nao, pna


def main(final_date):

    output_dir = f'{telnet_datadir}'

    indices = ['oni', 'pdo', 'atn-sst', 'ats-sst', 'atl-sst', 'sasdi', 'iobw', 'iod', 'ao', 'aao', 'psa1', 'psa2', 'nao', 'pna']
    
    final_season = datetime.strptime(final_date, '%Y-%m-%d') - pd.DateOffset(months=1)
    if final_season >= datetime.strptime('2024-12-01', '%Y-%m-%d'):
        time = pd.date_range('2025-01-01', final_season, freq='MS')
    else:
        time = pd.date_range('1941-01-01', final_season, freq='MS')
    indices_arr = np.full((len(time), len(indices)), np.nan)

    oni, pdo, atn, ats, atl, sasdi, iobw, iod, ao, aao, psa1, psa2, nao, pna = compute_climate_indices(final_season)
    indices_arr[:, :] = np.array([oni, pdo, atn, ats, atl, sasdi, iobw, iod, ao, aao, psa1, psa2, nao, pna]).T[-time.shape[0]:, :]

    df = pd.DataFrame(indices_arr, time, indices)
    if final_season >= datetime.strptime('2024-12-01', '%Y-%m-%d'):
        df.to_csv(f'{output_dir}/seasonal_climate_indices_2025-present.txt', ' ', float_format='%.03f', index_label=['time'])
    else:
        df.to_csv(f'{output_dir}/seasonal_climate_indices_1941-2024.txt', ' ', float_format='%.03f', index_label=['time'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute climate indices from ERA5 data')
    parser.add_argument('-fdate','--finaldate', help='Final download date in the format YYYYMM. Ex. 199310', required=True, default=None)
    args = parser.parse_args()
    final_date = f'{args.finaldate[:4]}-{args.finaldate[4:6]}-01'

    main(final_date)

