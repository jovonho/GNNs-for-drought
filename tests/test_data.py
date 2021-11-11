import numpy as np

from src.data import Dataset


def test_dataset_normalization():
    dataset = Dataset()

    _ = dataset.load_all_data()

    # an easy check is to make sure
    # the normalization worked for the static data
    static_data = dataset.cached_static_data
    for idx in range(static_data.shape[-1]):
        assert np.isclose(static_data[:, :, idx].mean(), 0, atol=0.1)
        assert np.isclose(np.std(static_data[:, :, idx]), 1, atol=0.1)


def test_filter_targets():
    dataset = Dataset()
    unfiltered_time_pairs = dataset.retrieve_date_tuples()

    # checks some time steps have been removed
    assert len(dataset.time_pairs) < len(unfiltered_time_pairs)
