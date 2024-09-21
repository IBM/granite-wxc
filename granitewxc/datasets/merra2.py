import numpy as np
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset

from PrithviWxC.dataloaders.merra2 import Merra2Dataset

class Merra2DownscaleDataset(Dataset):
    """For Climate downscaling Input data has low resolution, and target data has high resolution
    Further information on downscaling: https://link.springer.com/article/10.1007/s00382-022-06343-9#Abs1
    """

    def __init__(
        self,
        data_path_surface: str,
        data_path_vertical: str,
        output_vars: list[str],
        input_surface_vars: Optional[list[str]] = None,
        input_static_surface_vars: Optional[list[str]] = None,
        input_vertical_vars: Optional[list[str]] = None,
        input_levels: Optional[list[float]] = None,
        time_range: slice = None,
        climatology_path_surface: Optional[str] = None,
        climatology_path_vertical: Optional[str] = None,
        transforms: list[Callable] = [],
        n_input_timestamps = 1,      
    ):
        """
        Args:
            data_path: Path to data of a downscale model (high res data)
            n_input_timestamps: Number of time steps in input
            input_size_level: Number of levels in input
            input_size_lon: Number of lon in input
            input_size_lat: Number of lat in input
            input_surface_vars: input surface variables
            input_upper_level_vars: input upper level variables
        """

        self.input_surface_vars = input_surface_vars
        self.input_static_surface_vars = input_static_surface_vars
        self.input_vertical_vars = input_vertical_vars
        self.input_levels = input_levels if input_levels is not None else list()
        self.output_vars = output_vars # assume surface vars
        self.transforms = transforms

        self.surface_vars = input_surface_vars
        # for cases output parameter in not included in surface
        # maintains input order which is important because of how the scalers are fetched
        # won't need to maintain order if scalers are part of Dataset class 
        for out_var in self.output_vars:
            if out_var not in self.surface_vars:
                self.surface_vars.append(out_var)

        input_times = [int(timestamp) for timestamp in np.linspace(n_input_timestamps*-3, -3, n_input_timestamps)]

        self.dataset = Merra2Dataset(
            data_path_surface = data_path_surface,
            data_path_vertical = data_path_vertical,
            climatology_path_surface = climatology_path_surface,
            climatology_path_vertical = climatology_path_vertical,
            surface_vars = self.surface_vars,
            static_surface_vars = self.input_static_surface_vars,
            vertical_vars = self.input_vertical_vars,
            levels = self.input_levels,
            lead_times = [0],
            time_range = time_range,
            input_times = input_times,
        )

        self.n_input_timestamps = n_input_timestamps

        self.num_datapoints = self.dataset.__len__()

    @staticmethod
    def _subset_idx_map(x, x_sub) -> torch.Tensor:
        """
        e.g.
        x = ['t', 'u', 'v']
        x_sub = ['v', 'u']
        returns [2, 1] which is the index of target_vars in input_vars
        """

        return torch.tensor([x.index(var) for var in x_sub])


    def get_data(self, index) -> dict[torch.Tensor | int]:
        """
        Returns a dictionary with keys `sur_static`, `ulv_static`, `sur_vals`, `ulv_vals`, `sur_tars`, `ulv_tars`.
        Each individual component is a numpy array. They generally have the shape `feature, dim_0, dim_1, dim_2, ...`.
        Which dimensions are present depends on the particular tensor.
        The dimension ordering is time -> level -> lat -> lon.

        Args:
            index: Index of the sample returned.
        Returns:
            Dictionary with keys 'sur_static', 'sur_vals', 'sur_tars', 'ulv_static', 'ulv_vals', 'ulv_tars',
                'lead_time'. For each, the value is as follows:
            sur_static: Numpy array of shape (7, lat, lon). For each pixel (lat, lon),
                the first dimension indexes sin(lat), cos(lon), sin(lon), cos(doy), sin(doy), cos(hod), sin(hod).
                Where doy is the day of the year [1, 366] and hod the hour of the day [0, 23].
            sur_vals: Torch tensor of shape (time, parameter, lat, lon).
            sur_tars: Torch tensor of shape (parameter, lat, lon).
            ulv_static: Numpy array of shape (4, level, lat, lon). For each pixel (level, lat, lon),
                the first dimension indexes level, sin(lat), cos(lon), sin(lon).
            ulv_vals: Torch tensor of shape (time, parameter, level, lat, lon).
            ulv_tars: Torch tensor of shape (time, parameter, level, lat, lon).
            lead_time: Integer. Optional
        """
        data = self.dataset.__getitem__(index)

        x_sur = data['sur_vals'].index_select(index=self._subset_idx_map(self.surface_vars, self.input_surface_vars), dim=0) # parameters x time x lat x lon
        x_upl = data['ulv_vals'] # parameters x level x time x lat x lon

        static = data['sur_static']

        x = torch.cat(
            [
                x_sur.reshape(-1, x_sur.shape[-3], x_sur.shape[-2], x_sur.shape[-1]), #  parameters,  time, lat,  lon
                x_upl.reshape(-1, x_sur.shape[-3], x_upl.shape[-2], x_upl.shape[-1]), # parameters x level, time, lat,  lon
            ],
            dim=0,
        ) # parameters x level, time, lat,  lon

        x = x.permute(1, 0, 2, 3).reshape(-1, x_sur.shape[-2], x_sur.shape[-1]) # time x parameters x level,  lat,  lon

        y = data['sur_vals'].index_select( # parameters x time x lat x lon
            index=self._subset_idx_map(self.surface_vars, self.output_vars), dim=0
        )
        
        if self.n_input_timestamps > 1:
            # parameters, time, lat, lon -> parameters, 1, lat, lon
            y = y[:, -1:]

        y = y.reshape(
            -1, data['sur_vals'].shape[-2], data['sur_vals'].shape[-1]
        ) # parameters, lat, lon

        ret = {
            'x': x,
            'y': y,
            'static': static,
        }

        if 'sur_climate' in data:
            climate_sur = data['sur_climate'].index_select(index=self._subset_idx_map(self.surface_vars, self.input_surface_vars), dim=0) # parameters, lat, lon
            climate_upl = data['ulv_climate'] # parameters, level, lat, lon

            ret['climate_x'] = torch.cat(
                [
                    climate_sur,
                    climate_upl.reshape(-1, climate_upl.shape[-2], climate_upl.shape[-1]),
                ],
                dim=0,
            )

            ret['climate_y'] = data['sur_climate'].index_select(
                index=self._subset_idx_map(self.surface_vars, self.output_vars), dim=0
                ).reshape(
                    -1, data['sur_climate'].shape[-2], data['sur_climate'].shape[-1]
            )

        return ret
    
    
    def __getitem__(self, index) -> dict[torch.Tensor | int]:
        """
        Returns a dictionary with keys: [x, x_static, y, y_static, climate].
        Each individual component is a numpy array. They generally have the shape `feature, dim_0, dim_1, dim_2, ...`.
        Which dimensions are present depends on the particular tensor.
        The dimension ordering is time -> level -> lat -> lon.

        Args:
            index: Index of the sample returned.
        Returns:
            Dictionary with keys [x, x_static, y, y_static, climate],
        """

        sample = self.get_data(index)
            
        for t in self.transforms:
            sample = t(sample)
        
        return sample


    def __str__(self) -> str:
        return (
            f"DownscaleDataset("
            f"Recs: {len(self)}, "
            f"Dataset: {str(self.dataset)}, "
        )


    def __len__(self) -> int:
        return self.dataset.__len__()
        