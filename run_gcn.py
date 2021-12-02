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

import warnings

warnings.simplefilter("ignore", RuntimeWarning)


def get_datasets(val=False, concat_noisy_to_normal=False):

    if val:
        trainset = TrainValDataset(
            is_val=False,
            val_ratio=0.07,
            flatten=False,
            input_noise_scale=0.1,
            target_noise_scale=0.1,
        )

        valset = TrainValDataset(
            is_val=True,
            val_ratio=0.07,
            flatten=False,
            input_noise_scale=0.1,
            target_noise_scale=0.1,
        )
        val_size = len(valset)
    else:
        trainset = Dataset(
            is_test=False,
            flatten=False,
            input_noise_scale=0.1,
            target_noise_scale=0.1,
        )
        val_size = 0

    testset = Dataset(is_test=True, flatten=False)

    train_size = len(trainset)
    test_size = len(testset)

    # Optionally concatenate the normal and noisy data
    # to double the number of training samples.
    if concat_noisy_to_normal:
        if val:
            trainset_normal = TrainValDataset(is_val=False, val_ratio=0.07, flatten=False)
            valset_normal = TrainValDataset(is_val=True, val_ratio=0.07, flatten=False)
            trainset = ConcatDataset([trainset, trainset_normal])
            valset = ConcatDataset([valset, valset_normal])
            val_size += len(valset_normal)
        else:
            trainset_normal = Dataset(is_val=False, flatten=False)
            trainset = ConcatDataset([trainset, trainset_normal])
            valset = None

        train_size += len(trainset_normal)

    print(
        f"Num training samples:\t{train_size}\n"
        f"Num validation samples:\t{val_size}\n"
        f"Num test samples:\t{test_size}"
    )

    if concat_noisy_to_normal:
        # return trainset_normal to get the adjacency learning features from
        # TODO: change the design so we don't have to do this
        return trainset, valset, testset, trainset_normal
    else:
        return trainset, valset, testset, None


def explore_model_params(
    num_epochs=50, device="cpu", val=False, concat_noisy_to_normal=False, verbose=True
):
    t_device = torch.device(device if (torch.cuda.is_available() and "cuda" in device) else "cpu")

    if concat_noisy_to_normal:
        trainset, valset, testset, adj_features_set = get_datasets(
            val=True, concat_noisy_to_normal=concat_noisy_to_normal
        )
    else:
        trainset, valset, testset = get_datasets(concat_noisy_to_normal=concat_noisy_to_normal)

    # Fill in parameters to explore
    # Will automatically explore all combinations
    params = OrderedDict(
        lr=[0.01, 0.005, 0.001],
        hid_dim=[100, 150, 250, 350],
        num_layer=[2],
        batch_size=[8, 16],
        adj_learn_ts=[0, 25, 50, 100, 200, 300],
    )

    m = RunManager(val=val, verbose=verbose)

    for run in RunBuilder.get_runs(params):

        # Check previous runs, so as not to repeat them
        if m.check_run_already_done(run):
            print(f"Already ran {run}\nSkipping")
            continue

        trainloader = DataLoader(trainset, batch_size=run.batch_size)

        if val:
            valloader = DataLoader(valset, batch_size=run.batch_size)

        testloader = DataLoader(testset, batch_size=run.batch_size)

        if concat_noisy_to_normal:
            adj_learn_features = adj_features_set.get_adj_learning_features(
                num_timestamps=run.adj_learn_ts
            )
        else:
            adj_learn_features = trainset.get_adj_learning_features(
                num_timestamps=run.adj_learn_ts
            )

        model = GCN(
            7, run.hid_dim, run.num_layer, adj_learn_features=adj_learn_features, device=device
        )
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=run.lr)
        criterion = nn.MSELoss(reduction="mean")

        m.begin_run(run, model, trainloader)

        print(f"Running model on: {next(model.parameters()).device}")

        for _ in range(num_epochs):

            # Puts the model in training mode
            # and activates gradients
            m.begin_epoch()

            for _, (X, y_true, mask) in enumerate(trainloader):

                y_pred = model(X.to(t_device))
                y_true, y_pred = filter_preds_test_by_mask(y_pred, y_true.to(t_device), mask)
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

            if val:
                val_mse = 0
                val_r2 = []
                for _, (X, y_true, mask) in enumerate(valloader):

                    y_pred = model(X.to(t_device))
                    y_true, y_pred = filter_preds_test_by_mask(y_pred, y_true.to(t_device), mask)
                    # Multiply the MSE by each batch size, we will divide by n after
                    val_mse += mean_squared_error(y_true.data, y_pred.data) * X.shape[0]
                    val_r2.append(r2_score(y_true.data, y_pred.data))

                val_mse = val_mse / len(valset)
                val_r2 = np.mean(val_r2)
                m.end_epoch(val_mse, val_r2)
            else:
                m.end_epoch()

        # Evaluate model on test set
        test_mse = 0
        test_r2 = []
        for _, (X, y_true, mask) in enumerate(testloader):

            y_pred = model(X.to(t_device))
            y_true, y_pred = filter_preds_test_by_mask(y_pred, y_true.to(t_device), mask)

            test_mse += mean_squared_error(y_true.data, y_pred.data) * X.shape[0]
            test_r2.append(r2_score(y_true.data, y_pred.data))

        test_mse = round(test_mse / len(testset), 2)
        test_r2 = round(np.mean(test_r2), 2)

        # switch save_model to save models that have MSE < 1 and R2 < 1
        m.end_run(test_mse, test_r2, save_model=True)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    explore_model_params(num_epochs=1, device=device, val=False, concat_noisy_to_normal=False)
