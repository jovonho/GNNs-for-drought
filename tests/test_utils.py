import numpy as np
import torch

from src.utils import filter_preds_test_by_mask


def test_masking_np():

    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    mask = np.array([[True, False, False], [False, True, True]])

    f_a, f_b = filter_preds_test_by_mask(a, b, mask)
    assert (f_a == np.array([1, 5, 6])).all()
    assert (f_b == np.array([1, 5, 6])).all()


def test_masking_torch():

    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[True, False, False], [False, True, True]])

    f_a, f_b = filter_preds_test_by_mask(a, b, mask)
    assert (f_a == torch.tensor([1, 5, 6])).all()
    assert (f_b == torch.tensor([1, 5, 6])).all()
