import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from src.models.gcn import GCN
from collections import OrderedDict
from torch.utils.data import DataLoader, ConcatDataset
from src.models.mlp import TrainValDataset
from src.runutils import RunBuilder, RunManager
from sklearn.metrics import r2_score, mean_squared_error

from src.data import Dataset
from src.utils import filter_preds_test_by_mask
from src.config import MODELS_PATH

import warnings

warnings.simplefilter("ignore", RuntimeWarning)


def explore_model_params(num_epochs=50, device="cpu", concat_noisy_to_normal=False):

    trainset = TrainValDataset(
        is_val=False, val_ratio=0.07, flatten=False, input_noise_scale=0.1, target_noise_scale=0.1
    )
    valset = TrainValDataset(
        is_val=True, val_ratio=0.07, flatten=False, input_noise_scale=0.1, target_noise_scale=0.1
    )
    testset = Dataset(is_test=True, flatten=False)

    train_size = len(trainset)
    val_size = len(valset)
    test_size = len(testset)

    # Optionally concatenate the normal and noisy data together
    # By default we train and validate on
    if concat_noisy_to_normal:
        trainset_normal = TrainValDataset(is_val=False, val_ratio=0.07, flatten=False)
        valset_normal = TrainValDataset(is_val=True, val_ratio=0.07, flatten=False)
        trainset = ConcatDataset([trainset, trainset_normal])
        valset = ConcatDataset([valset, valset_normal])

        train_size += len(trainset_normal)
        val_size += len(valset_normal)

    print(
        f"Num training samples:\t{train_size}\n"
        f"Num Validation samples:\t{val_size}\n"
        f"Num test samples:\t{test_size}"
    )

    # Fill in parameters to explore
    # Will automatically explore all combinations
    params = OrderedDict(
        lr=[0.01, 0.005, 0.001],
        hid_dim=[100, 150, 250, 350],
        num_layer=[2],
        batch_size=[8, 16],
        adj_learn_ts=[0, 25, 50, 100, 200, 300],
    )

    m = RunManager()

    for run in RunBuilder.get_runs(params):

        # Check previous runs, so as not to repeat them
        if m.check_run_already_done(run):
            print(f"Already ran {run}\nSkipping")
            continue

        trainloader = DataLoader(trainset, batch_size=run.batch_size)
        valloader = DataLoader(valset, batch_size=run.batch_size)
        testloader = DataLoader(testset, batch_size=run.batch_size)

        adj_learn_features = trainset.get_adj_learning_features(num_timestamps=run.adj_learn_ts)

        model = GCN(
            7,
            run.hid_dim,
            run.num_layer,
            adj_learn_features=adj_learn_features,
        )

        optimizer = optim.Adam(model.parameters(), lr=run.lr)
        criterion = nn.MSELoss(reduction="mean")

        m.begin_run(run, model, trainloader, device)

        print(f"Running model on: {next(model.parameters()).device}")

        for _ in range(num_epochs):

            # Puts the model in training mode
            # and activates gradients
            m.begin_epoch()

            for i, (X, y_true, mask) in enumerate(trainloader):
                y_pred = model(X)
                y_true, y_pred = filter_preds_test_by_mask(y_pred, y_true, mask)
                loss = criterion(y_pred, y_true)
                batch_r2 = r2_score(y_true.data, y_pred.data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                m.track_r2(batch_r2)
                m.track_loss(loss)

            # Put the model in evaluation mode so weights don't get updated
            torch.no_grad()
            model.eval()

            # Validate
            val_r2 = []
            val_mse = 0
            for i, (X, y_true, mask) in enumerate(valloader):
                y_pred = model(X)
                y_true, y_pred = filter_preds_test_by_mask(y_pred, y_true, mask)
                val_r2.append(r2_score(y_true.data, y_pred.data))
                val_mse += mean_squared_error(y_true.data, y_pred.data) * X.shape[0]

            val_mse = val_mse / len(valset)
            val_r2 = np.mean(val_r2)
            num_edges = len(torch.nonzero(model.A.detach(), as_tuple=False))
            m.end_epoch(num_edges, val_mse, val_r2)

        # Evaluate model on test set
        test_mse = 0
        test_r2 = []
        for i, (X, y_true, mask) in enumerate(testloader):
            y_pred = model(X)
            y_true, y_pred = filter_preds_test_by_mask(y_pred, y_true, mask)
            test_r2.append(r2_score(y_true.data, np.nan_to_num(y_pred.data)))
            test_mse += mean_squared_error(y_true.data, y_pred.data) * X.shape[0]

        test_r2 = round(np.nan_to_num(np.mean(test_r2)), 2)
        test_mse = round(test_mse / len(testset), 2)
        print(f"Test Set:\n\tMSE {test_mse}\n\tR2 {test_r2}\n")

        if test_mse < 1 and test_r2 < 100:
            print(f"Test MSE {test_mse} < 1 and R2 {test_r2} < 100. Saving model.")
            MODELS_FILENAME = f"GCN-{num_epochs}ep-MSE={test_mse}-TestR2={test_r2}-{run}.pth"
            torch.save(
                model,
                MODELS_PATH / MODELS_FILENAME,
            )

        m.end_run(test_mse, test_r2)


if __name__ == "__main__":

    explore_model_params(num_epochs=50, device="cpu")
