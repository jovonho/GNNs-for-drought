from pathlib import Path
import xarray as xr
import numpy as np

from src.config import DATAFOLDER_PATH, TEST_YEARS, TARGET_DATASET

from typing import List, Tuple, Optional


class Dataset:
    """
    For now, this model will receive the previous timestep
    as input to predict VCI at the following timestep
    """

    def __init__(self, data_folder: Path = DATAFOLDER_PATH, is_test: bool = False) -> None:
        self.data_folder = data_folder
        self.is_test = is_test

        self.dynamic_datasets = self._retrieve_dynamic_interim_datasets()
        self.static_datasets = self._retrieve_static_interim_datasets()

        self.time_pairs = self.retrieve_date_tuples()

        self.cached_static_data: Optional[np.ndarray] = None
        self.cached_target_data: Optional[xr.Dataset] = None

    def __len__(self) -> int:
        return len(self.time_pairs)

    def _retrieve_dynamic_interim_datasets(self) -> List[str]:
        interim_folder = self.data_folder / "interim"
        foldernames = [x for x in interim_folder.glob("*processed")]
        assert len(foldernames) > 0
        # glob can behave strangely depending on the OS - sorting will ensure
        # consistent behaviour
        return sorted(foldernames)

    def _retrieve_static_interim_datasets(self) -> List[str]:
        interim_folder = self.data_folder / "interim/static"
        foldernames = [x for x in interim_folder.glob("*processed")]
        assert len(foldernames) > 0
        # glob can behave strangely depending on the OS - sorting will ensure
        # consistent behaviour
        return sorted(foldernames)

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

    def load_target_data_for_timestep(self, timestep: str) -> np.ndarray:
        if self.cached_target_data is None:
            self.cached_target_data = xr.open_dataset(
                self.data_folder / "interim" / TARGET_DATASET / "data_kenya.nc"
            )

        data_vars = list(self.cached_target_data.data_vars)
        assert len(data_vars) == 1

        return self.cached_target_data.sel(time=timestep)[data_vars[0]].values

    def load_dynamic_data_for_timestep(self, timestep: str) -> np.ndarray:
        arrays_list: List[np.ndarray] = []
        for dataset in self.dynamic_datasets:
            ds = xr.open_dataset(self.data_folder / "interim" / dataset / "data_kenya.nc")
            variables = sorted(list(ds.data_vars))

            ds_at_timestep = ds.sel(time=timestep)
            for data_var in variables:
                arrays_list.append(ds_at_timestep[data_var].values)
        return np.stack(arrays_list, axis=-1)

    def load_static_data(self) -> np.ndarray:
        if self.cached_static_data is None:
            # this stays constant for all input data, so it only needs to be loaded once
            arrays_list: List[np.ndarray] = []
            for dataset in self.static_datasets:
                ds = xr.open_dataset(
                    self.data_folder / "interim/static" / dataset / "data_kenya.nc"
                )
                variables = sorted(list(ds.data_vars))
                for data_var in variables:
                    arrays_list.append(ds[data_var].values)
            self.cached_static_data = np.stack(arrays_list, axis=-1)
        return self.cached_static_data

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:

        x_timestep, y_timestep = self.time_pairs[idx]

        dynamic_data = self.load_dynamic_data_for_timestep(x_timestep)
        static_data = self.load_static_data()

        x_data = np.concatenate([dynamic_data, static_data], axis=-1)
        target_data = self.load_target_data_for_timestep(y_timestep)

        # TODO: this is a very basic way of dealing with NaNs right now. It should
        # be improved
        x_data = np.nan_to_num(x_data)
        target_data = np.nan_to_num(target_data)

        # finally, flatten everything - our basic sklearn regressor
        # is going to expect a 2d input
        return x_data.flatten(), target_data.flatten()

    def load_all_data(self) -> np.ndarray:
        """
        Return two arrays, with the following shapes:
        x: [num_instances, num_features]
        y: [num_instances, num_targets]
        """
        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        for idx in range(len(self)):
            x, y = self[idx]
            x_list.append(x)
            y_list.append(y)

        return np.stack(x_list), np.stack(y_list)
