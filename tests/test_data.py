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


def test_gaussian_noise():
    # Check that two noisy datasets are not equal
    # i.e difference btw two should be all non-zero

    d1 = Dataset(input_noise_scale=0.2, target_noise_scale=0.2)
    d2 = Dataset(input_noise_scale=0.2, target_noise_scale=0.2)

    idx_rand = np.random.randint(1, len(d1))

    # Should be all non-zero
    # Check the first and a random line of the Xs
    assert (d1[0][0] - d2[0][0]).all()
    assert (d1[idx_rand][0] - d2[idx_rand][0]).all()

    # Check the first and a random line of the target
    assert (d1[0][1] - d2[0][1]).all()
    assert (d1[idx_rand][1] - d2[idx_rand][1]).all()


def test_no_gaussian_noise():
    # Check that two non-noisy datasets are equal
    # i.e difference btw two should be all zeros

    d1 = Dataset(input_noise_scale=0.0, target_noise_scale=0.0)
    d2 = Dataset(input_noise_scale=0.0, target_noise_scale=0.0)

    idx_rand = np.random.randint(1, len(d1))

    assert not (d1[0][0] - d2[0][0]).any()
    assert not (d1[idx_rand][0] - d2[idx_rand][0]).any()

    assert not np.nan_to_num(d1[0][1] - d2[0][1]).any()
    assert not np.nan_to_num(d1[idx_rand][1] - d2[idx_rand][1]).any()
