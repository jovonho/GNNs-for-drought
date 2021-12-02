import torch
import xarray as xr
import numpy as np
from pathlib import Path
from itertools import product

from src.config import DATAFOLDER_PATH, TEST_YEARS, TARGET_DATASET

from typing import Dict, List, Tuple, Optional


class Dataset:
    """
    For now, this model will receive the previous timestep
    as input to predict VCI at the following timestep
    """

    def __init__(
        self,
        data_folder: Path = DATAFOLDER_PATH,
        is_test: bool = False,
        flatten: bool = True,
        normalize_targets: bool = True,
        input_noise_scale: float = 0.0,
        target_noise_scale: float = 0.0,
    ) -> None:

        self.data_folder = data_folder
        self.is_test = is_test
        self.flatten = flatten
        self.normalize_targets = normalize_targets
        self.input_noise_scale = input_noise_scale
        self.target_noise_scale = target_noise_scale

        self.dynamic_datasets = self._retrieve_dynamic_interim_datasets()
        self.static_datasets = self._retrieve_static_interim_datasets()
        self.coordinates = self._get_coordinates()

        self.cached_static_data: Optional[np.ndarray] = None
        self.cached_target_data: Optional[xr.Dataset] = None
        self.cached_dynamic_means_and_stds: Dict[
            str, Tuple[float, float]
        ] = self._calculate_dynamic_means_and_stds()
        print(self.cached_dynamic_means_and_stds.keys())

        self.time_pairs = self.retrieve_date_tuples()
        self._filter_targets()

        self.cached_static_means_and_stds: Dict[str, Tuple[float, float]] = {}

    def __len__(self) -> int:
        return len(self.time_pairs)

    def _get_coordinates(self) -> List[Tuple]:
        ds = xr.open_dataset(
            self.data_folder / "interim/static" / self.static_datasets[0] / "data_kenya.nc"
        )
        return list(product(ds.indexes["lon"], ds.indexes["lat"]))

    def _retrieve_dynamic_interim_datasets(self) -> List[str]:
        interim_folder = self.data_folder / "interim"
        foldernames = [x for x in interim_folder.glob("*processed")]
        assert len(foldernames) > 0
        # glob can behave strangely depending on the OS - sorting will ensure
        # consistent behaviour
        return [x.name for x in sorted(foldernames)]

    def _retrieve_static_interim_datasets(self) -> List[str]:
        interim_folder = self.data_folder / "interim/static"
        foldernames = [x for x in interim_folder.glob("*processed")]
        assert len(foldernames) > 0
        # glob can behave strangely depending on the OS - sorting will ensure
        # consistent behaviour
        return [x.name for x in sorted(foldernames)]

    @staticmethod
    def _is_test_time(time_string: str) -> bool:
        return int(time_string[:4]) in TEST_YEARS

    def retrieve_date_tuples(self) -> List[Tuple[str, str]]:
        # This will be used to create the training data. Depending on whether
        # this is a test dataloader or not, we want to figure out which dates
        # are relevant to return as np.ndarrays.

        # We assume all datasets have even gaps.
        # However, not all datasets cover the same time range
        # so we need to find the common timerange between all datasets
        earliest_time = None
        latest_time = None
        common = np.array([], dtype=np.datetime64)

        for dataset in self.dynamic_datasets:
            ds = xr.open_dataset(self.data_folder / "interim" / dataset / "data_kenya.nc")

            ds_times = np.sort(ds.time.values)

            # Intialize values
            if not earliest_time:
                earliest_time = ds_times[0]
            if not latest_time:
                latest_time = ds_times[-1]
            if common.size == 0:
                common = ds_times

            # Case 1: The current dataset's earliest time is after the common range's start.
            # Remove all times occuring before this new earliest time from the common timerange.
            #
            # ds:             |-----------|
            # common:       |---------------|
            # new common:     |-------------|
            #
            if ds_times[0] > earliest_time:
                earliest_time = ds_times[0]
                common = common[common >= earliest_time]

            # Case 2: A dataset's latest time is before the common range's end
            # Remove all times occuring after this new latest time form the common time range.
            #
            # ds:             |-----------|
            # common:         |-------------|
            # new common:     |-----------|
            #
            if ds_times[-1] < latest_time:
                latest_time = ds_times[-1]
                common = common[common <= latest_time]

            # Tested with different loading orders and subsets of available datasets.
            #
            # The common timerange can only ever be reduced. Increasing it
            # on either side would mean the dataset you initialized it with did not cover
            # that same range, and thus it is not actually common to all datasets.
            #
            # If we initialize to the dataset with the largest timerange,
            # we will progressively reduce it as we encounter smaller timeranges on both ends.
            # If we initilialize to the smallest timerange, we already have the smallest
            # common timerange. Cases in between are reduced progressively on either sides.

        common = common.astype(str)
        if self.is_test:
            common = [x for x in common if self._is_test_time(x)]
        else:
            print(f"\nCommon time range of datasets: {common[0]} - {common[-1]}")
            common = [x for x in common if not self._is_test_time(x)]

        print(f"{'Test' if self.is_test else 'Train'} set time range {common[0]} - {common[-1]}")
        return [(common[idx - 1], common[idx]) for idx in range(1, len(common))]

    def load_target_data_for_timestep(self, timestep: str) -> Tuple[np.ndarray, np.ndarray]:
        if self.cached_target_data is None:
            self.cached_target_data = xr.open_dataset(
                self.data_folder / "interim" / TARGET_DATASET / "data_kenya.nc"
            )

        data_vars = list(self.cached_target_data.data_vars)
        assert len(data_vars) == 1

        target_values = self.cached_target_data.sel(time=timestep)[data_vars[0]].values

        if self.normalize_targets:
            identifier = f"{TARGET_DATASET}_{data_vars[0]}"
            target_mean, target_std = self.cached_dynamic_means_and_stds[identifier]

            target_values = (target_values - target_mean) / target_std

        gaussian_noise = np.random.normal(0, self.target_noise_scale, target_values.shape)
        target_values += gaussian_noise

        return target_values, ~np.isnan(target_values)

    def _filter_targets(self) -> None:
        indices_to_remove = []
        for idx in range(len(self)):
            _, y_timestep = self.time_pairs[idx]
            target_data, _ = self.load_target_data_for_timestep(y_timestep)
            if np.count_nonzero(np.isnan(target_data)) == target_data.size:
                indices_to_remove.append(idx)
        self.time_pairs = [
            val for idx, val in enumerate(self.time_pairs) if idx not in indices_to_remove
        ]

    def _calculate_dynamic_means_and_stds(self) -> Dict[str, Tuple[float, float]]:
        output_dict: Dict[str, Tuple[float, float]] = {}
        for dataset in self.dynamic_datasets:
            ds = xr.open_dataset(self.data_folder / "interim" / dataset / "data_kenya.nc")
            variables = sorted(list(ds.data_vars))

            for data_var in variables:
                variable = ds[data_var]
                var_label = f"{dataset}_{data_var}"
                variable_mean = np.nanmean(variable)
                variable_std = np.nanstd(variable)
                output_dict[var_label] = (
                    variable_mean,
                    variable_std,
                )
        return output_dict

    @property
    def variables_in_x(self) -> List[str]:
        output_vars: List[str] = []
        for dataset in self.dynamic_datasets:
            ds = xr.open_dataset(self.data_folder / "interim" / dataset / "data_kenya.nc")
            variables = sorted(list(ds.data_vars))
            output_vars.extend(variables)
        return output_vars

    def load_dynamic_data_for_timestep(self, timestep: str) -> np.ndarray:
        arrays_list: List[np.ndarray] = []

        for dataset in self.dynamic_datasets:
            ds = xr.open_dataset(self.data_folder / "interim" / dataset / "data_kenya.nc")
            variables = sorted(list(ds.data_vars))

            for data_var in variables:
                variable = ds[data_var]
                var_label = f"{dataset}_{data_var}"
                mean, std = self.cached_dynamic_means_and_stds[var_label]
                var_at_ts = np.nan_to_num(variable.sel(time=timestep).values, nan=mean)
                arrays_list.append((var_at_ts - mean) / std)

        return np.stack(arrays_list, axis=-1)

    def load_static_data(self) -> np.ndarray:
        if self.cached_static_data is None:
            # this stays constant for all input data, so it only needs to be loaded once
            arrays_list: List[np.ndarray] = []
            for dataset in self.static_datasets:
                ds = xr.open_dataset(
                    self.data_folder / "interim/static" / dataset / "data_kenya.nc"
                )
                # Cache a list of node coordinates for future reference
                if self.coordinates is None:
                    self.coordinates = list(product(ds.indexes["lon"], ds.indexes["lat"]))
                variables = sorted(list(ds.data_vars))
                for data_var in variables:
                    var_label = f"{dataset}_{data_var}"
                    if var_label not in self.cached_static_means_and_stds:
                        variable_mean = np.nanmean(ds[data_var].values)
                        variable_std = np.nanstd(ds[data_var].values)
                        self.cached_static_means_and_stds[var_label] = (
                            variable_mean,
                            variable_std,
                        )
                    mean, std = self.cached_static_means_and_stds[var_label]
                    normed_var = (np.nan_to_num(ds[data_var].values, nan=mean) - mean) / std

                    gaussian_noise = np.random.normal(0, self.input_noise_scale, normed_var.shape)
                    normed_var += gaussian_noise

                    arrays_list.append(normed_var)
            self.cached_static_data = np.stack(arrays_list, axis=-1)
        return self.cached_static_data

    @property
    def num_features(self) -> int:
        x, _, _ = self[0]
        return x.shape[0]

    @property
    def num_predictands(self) -> int:
        _, y, _ = self[0]
        return y.shape[0]

    def get_adj_learning_features(self, num_timestamps=344) -> np.ndarray:
        """
        Graphino returns normalized lat/lon + (2 features * 1200 obs) for each node
        for an total output size of 1345 * 2402.
        We return a 1575 * (num_timestamps * 7 features) matrix, with num_t = 344 by default to
        return a similar sized matrix.

        NOTE: In graphino the feature values are contiguous in the matrix
        whereas for us their are interleaved. This could be a source of
        differences in the fit of the learned adjacency matrix.
        """
        max_lat = np.max(self.coordinates)
        static_feats = np.array(
            [[lat / max_lat, (lon - 180) / 360] for lat, lon in self.coordinates]
        )
        # TODO: This is slow, might be faster to extract full columns from the xarray directly
        for idx in range(num_timestamps):
            x, _, _ = self[idx]
            # reshape to 1575 * 7
            static_feats = np.concatenate((static_feats, x), axis=1)

        return static_feats

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        x_timestep, y_timestep = self.time_pairs[idx]

        dynamic_data = self.load_dynamic_data_for_timestep(x_timestep)
        static_data = self.load_static_data()

        x_data = np.concatenate([dynamic_data, static_data], axis=-1)
        target_data, mask = self.load_target_data_for_timestep(y_timestep)

        # finally, flatten everything - our basic sklearn regressor
        # is going to expect a 2d input
        if self.flatten:
            return x_data.flatten(), target_data.flatten(), mask.flatten()
        else:
            dims = x_data.shape
            x_data = torch.as_tensor(
                x_data.reshape((dims[0] * dims[1], dims[2])), dtype=torch.float32
            )
            target_data = torch.as_tensor(
                target_data.reshape(dims[0] * dims[1]), dtype=torch.float32
            )
            mask_data = torch.as_tensor(mask.reshape(dims[0] * dims[1]), dtype=torch.bool)
            return x_data, target_data, mask_data

    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return two arrays, with the following shapes:
        x: [num_instances, num_features]
        y: [num_instances, num_targets]
        """
        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []
        for idx in range(len(self)):
            x, y, mask = self[idx]
            x_list.append(x)
            y_list.append(y)
            mask_list.append(mask)

        return np.stack(x_list), np.stack(y_list), np.stack(mask_list)


# TODO: When running this file directly, we get an error saying No module named src
# But if we remove it, we get errors when running benchmark_*.py files
if __name__ == "__main__":
    labels = xr.open_dataset(Path("./data/interim") / TARGET_DATASET / "data_kenya.nc")

    print(labels.sel(time="1985-02-28").VCI)
