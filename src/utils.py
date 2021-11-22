import numpy as np
import torch

from typing import Union, Tuple


def filter_preds_test_by_mask(
    y: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:

    # return y and yhat masked by the predictions

    assert type(y) == type(y_pred) == type(mask)
    if isinstance(y, torch.Tensor):
        return torch.masked_select(y, mask), torch.masked_select(y_pred, mask)
    else:
        assert isinstance(y, np.ndarray)
        return y[mask], y_pred[mask]
