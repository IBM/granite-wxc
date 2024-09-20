import os
from functools import partial
from typing import Optional
import logging
from packaging.version import parse as parse_version

import numpy as np
import torch
import xarray as xr
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if parse_version(xr.__version__) >= parse_version('2023.11.0'):
    XARRAY_TO_DATAARRAY = True
else:
    XARRAY_TO_DATAARRAY = False

from granitewxc.utils.cache import Cache
from granitewxc.utils.distributed import get_local_rank
from granitewxc.utils import distributed
from granitewxc.utils.config import ExperimentConfig
from granitewxc.datasets.merra2 import Merra2DownscaleDataset

def torch_interpolate_3d(x, size=None, scale_factor=None, mode='bilinear', align_corners=None):
    x = torch.unsqueeze(x, 0)
    x = F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    x = torch.squeeze(x, 0)
    return x


def crop(sample: dict[str, torch.Tensor], crop_lat: list[int], crop_lon: list[int]):

    for k, v in sample.items():
        sample[k] = v[..., crop_lat[0]:crop_lat[1], crop_lon[0]:crop_lon[1]]
    
    return sample


def transform(sample, size=(35, 35)):

    ret = dict()

    retain_keys = dict(
        y='y',
        static='static_y',
        climate_y='climate_y',
    )

    for k_orig, k_new in retain_keys.items():
        if k_orig in sample:
            ret[k_new] = sample[k_orig]

    interpolate_keys = dict(
        x='x',
        static='static_x',
        climate_x='climate_x'
    )

    for k_orig, k_new in interpolate_keys.items():
        if k_orig in sample:
            ret[k_new] = torch_interpolate_3d(
                sample[k_orig], size=size, mode='bilinear', align_corners=True
            )

    return ret


def smoothen(sample):
    padding_mode = 'replicate'

    for k in ['x', 'static_x', 'climate_x']:
        if k in sample:
            n_dims = sample[k].shape[0]

            kernel = torch.ones((3, 3))
            kernel = kernel / kernel.sum()
            kernel = torch.einsum(
                'ij,kl->ijkl', 
                torch.eye(n_dims), kernel
            )

            sample[k] = F.conv2d(
                F.pad(sample[k], (1, 1, 1, 1), mode=padding_mode),
                kernel, padding='valid'
            )

    return sample


def preproc_downscale_merra2(
        batch: list[dict],
        input_padding: dict[tuple[int]],
        target_padding: dict[tuple[int]],
    ) -> dict:
    """
    inputs and targets can have different parameters 
    Args:
        batch: List of training samples. Each sample should be a dictionary with keys 'sur_static', 'sur_vals',
            'sur_tars', 'ulv_static', 'ulv_vals', 'ulv_tars', 'lead_time'.
            The tensors have shape:
                sur_static: Numpy array of shape (3, lat, lon). For each pixel (lat, lon),
                            the first dimension indexes sin(lat), cos(lon), sin(lon).
                sur_vals: Torch tensor of shape (parameter, time, lat, lon).
                sur_tars: Torch tensor of shape (parameter, time, lat, lon).
                ulv_static: Numpy array of shape (4, level, lat, lon). For each pixel (level, lat, lon),
                            the first dimension indexes level, sin(lat), cos(lon), sin(lon).
                ulv_vals: Torch tensor of shape (parameter, level, time, lat, lon).
                ulv_tars: Torch tensor of shape (parameter, level, time, lat, lon).
                sur_climate: Torch tensor of shape (parameter, lat, lon)
                ulv_climate: Torch tensor of shape (parameter, level, lat, lon)
                lead_time: Integer.
        input_padding: Padding values for input data. Dictionary with keys 'level', 'lat', 'lon. For each,
            the value is a tuple of length 2 indicating padding at the start and end of the relevant dimension.
        target_padding: Padding values for target data (same structure as *input_padding*)
    Returns:
        Dictionary with keys 'x', 'y', 'lead_time' and 'static'. All are torch tensors. Shapes are as follows:
            x: batch, time, parameter, lat, lon
            y: batch, parameter, lat, lon
            static: batch, parameter, lat, lon
            lead_time: batch
        Here, for x and y, 'parameter' is [surface parameter, upper level parameter x level]
    """
    if not set(batch[0].keys()).issuperset(
        ['sur_static', 'sur_vals', 'ulv_static', 'ulv_vals', 'lead_time']
    ):
        raise ValueError('Missing essential keys.')
    if not set(
        [
            'sur_static',
            'sur_vals',
            'sur_tars',
            'ulv_static',
            'ulv_vals',
            'ulv_tars',
            'sur_climate',
            'ulv_climate',
            'lead_time',
        ]
    ).issuperset(batch[0].keys()):
        raise ValueError('Unexpected keys in batch.')
    if set(batch[0].keys()).intersection(set(['ulv_tars', 'sur_tars'])) == set():
        raise ValueError(f"Need or both of {set('ulv_tars', 'sur_tars')}")

    # Bring all tensors from the batch into a single tensor
    upl_x = torch.empty((len(batch), *batch[0]['ulv_vals'].shape))
    sur_x = torch.empty((len(batch), *batch[0]['sur_vals'].shape))

    upl_y = torch.empty((len(batch), *batch[0]['ulv_tars'].shape)) if 'uvl_tars' in batch[0].keys() else None
    sur_y = torch.empty((len(batch), *batch[0]['sur_tars'].shape)) if 'sur_tars' in batch[0].keys() else None

    sur_sta = torch.empty((len(batch), *batch[0]['sur_static'].shape))
    upl_sta = torch.empty((len(batch), *batch[0]['ulv_static'].shape))

    lead_time = torch.empty((len(batch),), dtype=torch.float32)

    for i, rec in enumerate(batch):
        sur_x[i] = torch.Tensor(rec['sur_vals'])
        upl_x[i] = torch.Tensor(rec['ulv_vals'])

        if sur_y is not None: sur_y[i] = torch.Tensor(rec['sur_tars'])
        if upl_y is not None: upl_y[i] = torch.Tensor(rec['ulv_tars'])

        sur_sta[i] = torch.Tensor(rec['sur_static'])
        upl_sta[i] = torch.Tensor(rec['ulv_static'])

        lead_time[i] = rec['lead_time']

    upl_x = upl_x.permute((0, 3, 1, 2, 4, 5))  # (batch, parameter, level, time, lat, lon) -> (batch, time, parameter, level, lat, lon)
    upl_x = upl_x.flatten(1, 2)  # [batch, time, parameter, lat_low_res, lon_low_res] -> [batch, time x parameter, lat_low_res, lon_low_res]
    if upl_y is not None:
        upl_y = upl_y.permute((0, 3, 1, 2, 4, 5))
        upl_y = upl_y.flatten(1, 2)

    sur_x = sur_x.permute((0, 2, 1, 3, 4))  # (batch, parameter, time, lat, lon) -> (batch, time, parameter, lat, lon)
    sur_x = sur_x.flatten(1, 2)  # [batch, time, parameter, lat_low_res, lon_low_res] -> [batch, time x parameter, lat_low_res, lon_low_res]
    if sur_y is not None:
        sur_y = sur_y.permute((0, 2, 1, 3, 4))
        sur_y = sur_y.flatten(1, 2)

    # Pad
    input_padding_2d = (*input_padding['lon'], *input_padding['lat'])
    input_padding_3d = (*input_padding['lon'], *input_padding['lat'], *input_padding['level'])

    target_padding_2d = (*target_padding['lon'], *target_padding['lat'])
    target_padding_3d = (*target_padding['lon'], *target_padding['lat'], *target_padding['level'])

    sur_x = torch.nn.functional.pad(sur_x, input_padding_2d, mode='constant', value=0)
    upl_x = torch.nn.functional.pad(upl_x, input_padding_3d, mode='constant', value=0)
    if sur_y is not None: sur_y = torch.nn.functional.pad(sur_y, target_padding_2d, mode='constant', value=0)
    if upl_y is not None: upl_y = torch.nn.functional.pad(upl_y, target_padding_3d, mode='constant', value=0)
    sur_sta = torch.nn.functional.pad(sur_sta, target_padding_2d, mode='constant', value=0)
    upl_sta = torch.nn.functional.pad(upl_sta, target_padding_3d, mode='constant', value=0)

    # We stack along the combined parameter x level dimension
    x = torch.cat(
        [
            sur_x,
            upl_x.reshape(*upl_x.shape[:1], upl_x.shape[1] * upl_x.shape[2], *upl_x.shape[3:]),
        ],
        dim=1,
    )

    if upl_y is not None and sur_y is not None:
        y = torch.cat(
            [
                sur_y,
                upl_y.reshape(upl_y.shape[0], upl_y.shape[1] * upl_y.shape[2], *upl_y.shape[3:]),
            ],
            dim=1,
        )
    elif upl_y is None:
        y = sur_y
    elif sur_y is None:
        y = upl_y.reshape(upl_y.shape[0], upl_y.shape[1] * upl_y.shape[2], *upl_y.shape[3:])
    else:  # should not reach this part of the code
        raise ValueError("No targets in data")

    return_value = {
        'x': x,
        'y': y,
        'lead_time': lead_time,
        'static': sur_sta,
    }

    return return_value


def get_dataloaders_merra2(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, Optional[Cache], Optional[Cache]]:
    """
    Args:
        config: Experiment configuration. Contains configuration parameters for model.
    Returns:
        Tuple of data loaders: (training loader, validation loader).
    """

    ds_kwargs = dict(
        data_path_surface = config.data.data_path_surface,
        data_path_vertical = config.data.data_path_vertical,
        climatology_path_surface = config.data.climatology_path_surface,
        climatology_path_vertical = config.data.climatology_path_vertical,
        input_surface_vars = config.data.input_surface_vars,
        input_static_surface_vars = config.data.input_static_surface_vars,
        input_vertical_vars = config.data.input_vertical_vars,
        input_levels = config.data.input_levels,
        n_input_timestamps = config.data.n_input_timestamps,
        output_vars=config.data.output_vars,
        transforms=_get_transforms(config),
    )

    train_dataset = Merra2DownscaleDataset(
        time_range=(config.data.train_time_range_start, config.data.train_time_range_end),
        **ds_kwargs
    )

    valid_dataset = Merra2DownscaleDataset(
        time_range=(config.data.val_time_range_start, config.data.val_time_range_end),
        **ds_kwargs
    )

    if config.data.cache:
        # We might have to adjust the cache dir ...
        pbs_job_id = os.environ.get('PBS_JOBID')
        if pbs_job_id is not None:
            config.data.cache_dir = os.path.join(config.data.cache_dir, pbs_job_id)

        cache_dir_train = os.path.join(config.data.cache_dir, 'train')
        cache_dir_valid = os.path.join(config.data.cache_dir, 'valid')
        os.makedirs(cache_dir_train, exist_ok=True)
        os.makedirs(cache_dir_valid, exist_ok=True)

        train_cache = Cache(
            train_dataset,
            cache_dir_train,
            cache_size=config.data.cache_size_train,
            n_workers=config.data.cache_num_workers,
            inactive=get_local_rank() != 0,
            clean_up=True,
        )
        valid_cache = Cache(
            valid_dataset,
            cache_dir_valid,
            cache_size=config.data.cache_size_valid,
            n_workers=config.data.cache_num_workers,
            inactive=get_local_rank() != 0,
            clean_up=True,
        )
        train_cache.start()
        valid_cache.start()

        if distributed.is_main_process():
            print(f"Initialized {config.data.cache_dir} as cache.")
    else:
        train_cache = None
        valid_cache = None

    dl_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.dl_num_workers,
        prefetch_factor=config.dl_prefetch_size,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    if config.data.cache:
        train_loader = DataLoader(dataset=train_cache.cached_dataset, **dl_kwargs)
        valid_loader = DataLoader(dataset=valid_cache.cached_dataset, **dl_kwargs)
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True),
            **dl_kwargs,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            sampler=DistributedSampler(valid_dataset, shuffle=False, drop_last=False),
            **dl_kwargs,
        )

    if distributed.is_main_process():
        print(f"--> Training batches: {len(train_loader):,.0f}")
        print(f"--> Validation batches: {len(valid_loader):,.0f}")
        print(f"--> Training samples: {len(train_dataset):,.0f}")
        print(f"--> Validation samples: {len(valid_dataset):,.0f}")

    return (
        train_loader,
        valid_loader,
        train_cache,
        valid_cache,
    )


def _get_transforms(config: ExperimentConfig):

    if 'crop_lat' in config.data.__dict__:
        crop_lat = [config.data.crop_lat[0], -config.data.crop_lat[1]]
    else:
        crop_lat = [0, None]

    if 'crop_lon' in config.data.__dict__:
        crop_lon = [config.data.crop_lon[0], -config.data.crop_lon[1]]
    else:
        crop_lon = [0, None]

    transforms = [
        partial(crop, crop_lat=crop_lat, crop_lon=crop_lon),
        partial(transform, size=(config.data.input_size_lat, config.data.input_size_lon)),
    ]
                  

    if config.data.__dict__.get('apply_smoothen', False):
        transforms.append(smoothen)

    return transforms


def get_dataloaders(config: ExperimentConfig, logger: logging.Logger=None) -> tuple[DataLoader, DataLoader]:
    """
    Calls one of:
      get_dataloaders_merra2
      get_dataloaders_cordex
    """
    if config.data.type == 'merra2':
        return get_dataloaders_merra2(config)
    else:
        raise ValueError(f'{config.data.type} is not a valid config.data.type')


def assemble_input_scalers(config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For merra2 based dataset where surface and upper level variables are specified separately
    """
    _kwargs_open_dataset = {'chunks' : None, 'cache' : True, 'engine' : 'h5netcdf'}

    with (
        xr.open_dataset(
            config.model.input_scalers_surface_path,
            **_kwargs_open_dataset
        ) as musigma_surface,
        xr.open_dataset(
            config.model.input_scalers_vertical_path,
            **_kwargs_open_dataset
        ) as musigma_vertical
    ):
        musigma_surface.load()
        musigma_vertical.load()
        
        mu_surface = musigma_surface[config.data.input_surface_vars].sel(statistic='mu')
        sigma_surface = musigma_surface[config.data.input_surface_vars].sel(statistic='sigma')
        mu_vertical = musigma_vertical[config.data.input_vertical_vars].sel(statistic='mu', lev=config.data.input_levels)
        sigma_vertical = musigma_vertical[config.data.input_vertical_vars].sel(statistic='sigma', lev=config.data.input_levels)

        if XARRAY_TO_DATAARRAY:
            mu_surface = mu_surface.to_dataarray(dim='parameter')
            sigma_surface = sigma_surface.to_dataarray(dim='parameter')
            mu_vertical = mu_vertical.to_dataarray(dim='parameter')
            sigma_vertical = sigma_vertical.to_dataarray(dim='parameter')
        else:
            mu_surface = mu_surface.to_array(dim='parameter')
            sigma_surface = sigma_surface.to_array(dim='parameter')
            mu_vertical = mu_vertical.to_array(dim='parameter')
            sigma_vertical = sigma_vertical.to_array(dim='parameter')
        
        mu = torch.cat(
            (
                torch.from_numpy(mu_surface.values),
                torch.from_numpy(mu_vertical.transpose('parameter', 'lev').values.reshape(-1)),
            ), dim=0
        ).to(dtype=torch.float32)
        
        sigma = torch.cat(
            (
                torch.from_numpy(sigma_surface.values),
                torch.from_numpy(sigma_vertical.transpose('parameter', 'lev').values.reshape(-1)),
            ), dim=0
        ).to(dtype=torch.float32)

        sigma = torch.clamp(sigma, 1e-4, 1e4)

    return mu, sigma


def assemble_target_scalers(config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For merra2 based dataset where surface and upper level variables are specified separately
    """
    _kwargs_open_dataset = {'chunks' : None, 'cache' : True, 'engine' : 'h5netcdf'}

    with (
        xr.open_dataset(
            config.model.input_scalers_surface_path,
            **_kwargs_open_dataset
        ) as musigma_surface,
    ):
        musigma_surface.load()
        
        mu_surface = musigma_surface[config.data.output_vars].sel(statistic='mu')
        sigma_surface = musigma_surface[config.data.output_vars].sel(statistic='sigma')

        if XARRAY_TO_DATAARRAY:
            mu_surface = mu_surface.to_dataarray(dim='parameter')
            sigma_surface = sigma_surface.to_dataarray(dim='parameter')
        else:
            mu_surface = mu_surface.to_array(dim='parameter')
            sigma_surface = sigma_surface.to_array(dim='parameter')
        
        mu = torch.from_numpy(mu_surface.values).to(dtype=torch.float32)
        sigma = torch.from_numpy(sigma_surface.values).to(dtype=torch.float32)
        sigma = torch.sqrt(torch.clamp(sigma**2, 1e-8, 1e6))

        return mu, sigma


def assemble_static_input_scalers(config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    _kwargs_open_dataset = {'chunks' : None, 'cache' : True, 'engine' : 'h5netcdf'}

    with xr.open_dataset(
            config.model.input_scalers_surface_path,
            **_kwargs_open_dataset
        ) as musigma_surface:
        musigma_surface.load()
        
        mu_surface = musigma_surface[config.data.input_static_surface_vars].sel(statistic='mu')
        sigma_surface = musigma_surface[config.data.input_static_surface_vars].sel(statistic='sigma')

        if XARRAY_TO_DATAARRAY:
            mu_surface = mu_surface.to_dataarray(dim='parameter')
            sigma_surface = sigma_surface.to_dataarray(dim='parameter')
        else:
            mu_surface = mu_surface.to_array(dim='parameter')
            sigma_surface = sigma_surface.to_array(dim='parameter')
        
        mu = torch.cat(
            (
                torch.zeros((config.model.num_static_channels,), dtype=torch.float32),
                torch.from_numpy(mu_surface.values),
            ), dim=0
        ).to(dtype=torch.float32)
        
        sigma = torch.cat(
            (
                torch.ones((config.model.num_static_channels,), dtype=torch.float32),
                torch.from_numpy(sigma_surface.values),
            ), dim=0
        ).to(dtype=torch.float32)

        sigma = torch.clamp(sigma, 1e-4, 1e4)

        return mu, sigma


def assemble_output_scalers(config: ExperimentConfig) -> torch.Tensor:
    _kwargs_open_dataset = {'chunks' : None, 'cache' : True, 'engine' : 'h5netcdf'}

    if config.model.residual == 'none':
        _, sigma = assemble_input_scalers(config)
        variances = sigma**2
        return variances

    with (
        xr.open_dataset(
            config.model.output_scalers_surface_path,
            **_kwargs_open_dataset
        ) as scaler_surface,
        xr.open_dataset(
            config.model.output_scalers_vertical_path,
            **_kwargs_open_dataset
        ) as scaler_vertical
    ):
        scaler_surface = scaler_surface[config.data.input_surface_vars]
        scaler_vertical = scaler_vertical[config.data.input_vertical_vars].sel(lev=config.data.input_levels)
        
        if XARRAY_TO_DATAARRAY:
            scaler_surface = scaler_surface.to_dataarray(dim='parameter')
            scaler_vertical = scaler_vertical.to_dataarray(dim='parameter')
        else:
            scaler_surface = scaler_surface.to_array(dim='parameter')
            scaler_vertical = scaler_vertical.to_array(dim='parameter')

        variances = torch.cat(
            (
                torch.from_numpy(scaler_surface.values),
                torch.from_numpy(scaler_vertical.transpose('parameter', 'lev').values.reshape(-1)),
            ),
            dim=0
        ).to(dtype=torch.float32)

    # Looking through the numbers, we have values as extreme as 1e-59 and 7e6.
    variances = torch.clamp(variances, 1e-7, 1e7)

    return variances


def assemble_climate_scalers(config) -> torch.Tensor:
    _kwargs_open_dataset = {'chunks' : None, 'cache' : True, 'engine' : 'h5netcdf'}

    with (
        xr.open_dataset(
            config.model.output_scalers_surface_path,
            **_kwargs_open_dataset
        ) as scaler_surface
    ):
        scaler_surface = scaler_surface[config.data.output_vars]

        if XARRAY_TO_DATAARRAY:
            scaler_surface = scaler_surface.to_dataarray(dim='parameter')
        else:
            scaler_surface = scaler_surface.to_array(dim='parameter')

        variances = torch.from_numpy(scaler_surface.values).to(dtype=torch.float32)

    # Looking through the numbers, we have values as extreme as 1e-59 and 7e6.
    variances = torch.clamp(variances, 1e-7, 1e7)

    return variances


def get_output_channels(config):
    """ Determines number of output channels based on user configuration. """
    out_channels = 0
    try:
        out_channels += len(config.data.target_surface_vars)
    except:
        print("Output does not contain surface variables")
    try:
        out_channels += (len(config.data.target_levels) * len(config.data.target_upper_level_vars))
    except:
        print("Output does not contain upper level variables")

    out_channels *= config.data.target_size_time

    return out_channels
