"""
VCI is highly autoregressive. A difficult to
beat baseline is to predict last month's VCI as
this month's VCI.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from src.data import Dataset
from src.utils import filter_preds_test_by_mask


def main():

    testset = Dataset(is_test=True, flatten=False)
    vci_index = testset.variables_in_x.index("VCI")

    mse = []
    r2 = []
    for _, (X, y_true, mask) in enumerate(testset):
        preds = X[:, vci_index]
        y_true, preds = filter_preds_test_by_mask(preds, y_true, mask)
        r2.append(r2_score(y_true.data, preds.data))
        mse.append(mean_squared_error(y_true.data, preds.data))

    r2 = np.mean(r2)
    mse = np.mean(mse)
    print(f"Test Set:\n\tMSE {mse}\n\tR2 {r2}")


if __name__ == "__main__":
    main()
