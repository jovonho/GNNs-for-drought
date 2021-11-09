import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from src.models.gcn import GCN
from collections import OrderedDict
from torch.utils.data import DataLoader
from src.runutils import RunBuilder, RunManager
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score, mean_squared_error

from src.data import Dataset
from src.config import MODELS_PATH, RUNS_PATH

import warnings

warnings.simplefilter("ignore", RuntimeWarning)


def main():

    t1 = time.time()

    trainset = Dataset(is_test=False, flatten=False)
    testset = Dataset(is_test=True, flatten=False)

    num_samples = len(trainset)

    # We need to get a matrix of dimensions #nodes * #number of features
    adj_learn_features = trainset.get_adj_learning_features()
    print(f"\nAdjacency learning features {adj_learn_features.shape}")

    lr = 0.005
    epochs = 20
    device = "cpu"
    batch_size = 32

    # Num workers doesn't work with these
    trainloader = DataLoader(trainset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    hid_dim = 250
    num_layer = 2
    model = GCN(7, hid_dim, num_layer, adj_learn_features=adj_learn_features)
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")

    comment = f" batch={batch_size} lr={lr} hid_dim={hid_dim} layers={num_layer}"
    tb = SummaryWriter(comment=comment)

    print("Training a GCN\n")

    log_iter = (num_samples // batch_size) // 4

    for epoch in range(epochs):

        epoch_loss = 0
        epoch_r2 = []
        progress_loss = 0
        progress_r2 = []

        for i, batch in enumerate(trainloader):
            X, y_true = batch

            y_pred = model(X)
            loss = criterion(y_pred, y_true)
            batch_loss = loss.item() * batch_size

            epoch_loss += batch_loss
            progress_loss += batch_loss

            batch_r2 = r2_score(y_true.data, np.nan_to_num(y_pred.data))
            epoch_r2.append(batch_r2)
            progress_r2.append(batch_r2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i != 0 and i % log_iter == 0:
                print(
                    f"Iteration {i}:"
                    f"\n\tloss (mse):\t{progress_loss / ((i+1) * batch_size)}"
                    f"\n\tR2 score:\t{np.mean(progress_r2)}"
                )

        epoch_r2 = np.mean(epoch_r2)
        num_edges = len(torch.nonzero(model.A.detach(), as_tuple=False))
        print(
            "---------------------------------------------------"
            f"\t\nEpoch {epoch}:\n\ttotal MSE:\t{epoch_loss}\n\t"
            f"epoch r2:\t{epoch_r2}\n\tnum edges:\t{num_edges}\n"
        )

        tb.add_scalar("MSE", epoch_loss, epoch)
        tb.add_scalar("R2", epoch_r2, epoch)
        tb.add_scalar("Num Edges", num_edges, epoch)

        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f"{name}.grad", weight.grad, epoch)

    print(f"Running time: {time.time() - t1} s")
    tb.flush()
    tb.close()

    MODEL_FILENAME = f"GCN-{epochs}epochs-{lr}lr-{batch_size}bs-{hid_dim}hd-{num_layer}L.pth"
    torch.save(model, MODELS_PATH / MODEL_FILENAME)

    # Evaluate on test set
    model.eval()
    mse = 0
    batch_r2 = 0
    for i, (X, y_true) in enumerate(testloader):
        preds = model(X)
        batch_r2 += r2_score(y_true.data, preds.data)
        mse += mean_squared_error(y_true.data, preds.data)

    batch_r2 = np.mean(batch_r2)
    print(f"Test Set:\n\tMSE {mse}\n\tR2 {batch_r2}")


def get_optimizer(optim_str, model_params, lr):
    """
    Seems like SGD is out of style nowadays and Adam is almost always better
    """
    optimizers = {
        "Adam": optim.Adam(model_params, lr=lr),
        "SGD": optim.SGD(model_params, lr=lr, weight_decay=1e-6, momentum=0.9, nesterov=True),
    }
    return optimizers[optim_str]


def explore_model_params(num_epochs=50):

    device = "cpu"

    trainset = Dataset(is_test=False, flatten=False)
    # num_samples = len(trainset)
    adj_learn_features = trainset.get_adj_learning_features()

    # Fill in parameters to explore
    # Will automatically explore all combinations
    params = OrderedDict(
        lr=[0.01, 0.005, 0.001],
        batch_size=[4, 8, 16, 32, 64],
        hid_dim=[100, 150, 250, 350],
        out_dim=[100, 150, 50],
        adj_learn_dim=[50, 100, 25],
        num_layer=[2, 3],
        optimizer=["Adam"],
    )
    m = RunManager()

    max_test_mse = 0

    import os
    import re

    # Check runs that already ran
    previous_runs = os.listdir(RUNS_PATH)

    for run in RunBuilder.get_runs(params):

        run_str = re.escape(run.__str__())
        r = re.compile(rf".*{run_str}")

        matches = list(filter(r.match, previous_runs))

        if len(matches) > 0:
            # Run already done
            print(f"Already ran {run}\nSkipping")
            continue

        # Make it so we get 4 intermediate print outs per epoch
        num_samples = len(trainset)
        log_iter = (num_samples // run.batch_size) // 4

        model = GCN(
            7,
            run.hid_dim,
            run.out_dim,
            run.num_layer,
            adj_learn_features=adj_learn_features,
            adj_learn_dim=run.adj_learn_dim,
        )
        model = model.to(device)
        model.train()
        trainloader = DataLoader(trainset, batch_size=run.batch_size)
        optimizer = optim.Adam(model.parameters(), lr=run.lr)
        criterion = nn.MSELoss(reduction="mean")

        m.begin_run(run, model, trainloader)

        for _ in range(num_epochs):
            progress_loss = 0
            progress_r2 = []

            m.begin_epoch()

            for i, (X, y_true) in enumerate(trainloader):

                y_pred = model(X)

                loss = criterion(y_pred, y_true)
                # Was getting some errors
                batch_r2 = r2_score(y_true.data, np.nan_to_num(y_pred.data))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                m.track_r2(batch_r2)
                m.track_loss(loss)

                progress_loss += loss.item() * run.batch_size
                progress_r2.append(batch_r2)

                if i != 0 and i % log_iter == 0:
                    print(
                        f"Iteration {i}:"
                        f"\n\tloss (mse):\t{progress_loss / ((i+1) * run.batch_size)}"
                        f"\n\tR2 score:\t{np.mean(progress_r2)}"
                    )

            num_edges = len(torch.nonzero(model.A.detach(), as_tuple=False))
            m.end_epoch(num_edges)

        # Evaluate model on test set
        model.eval()
        testset = Dataset(is_test=True, flatten=False)
        testloader = DataLoader(testset, batch_size=run.batch_size)
        test_num_samples = len(testset)

        test_mse = 0
        test_r2 = []
        for i, (X, y_true) in enumerate(testloader):
            preds = model(X)
            test_r2.append(r2_score(y_true.data, np.nan_to_num(preds.data)))
            test_mse += mean_squared_error(y_true.data, preds.data) * run.batch_size

        test_r2 = round(np.nan_to_num(np.mean(test_r2)), 2)
        test_mse = round(test_mse / test_num_samples, 2)
        print(f"Test Set:\n\tMSE {test_mse}\n\tR2 {test_r2}\n")

        if test_mse > max_test_mse and test_mse < 500:
            print(f"New High Score! ({test_mse} vs {max_test_mse}\nSaving model")
            max_test_mse = test_mse
            MODELS_FILENAME = f"GCN-{num_epochs}epochs-TestMSE={test_mse}-{run}.pth"
            torch.save(
                model,
                MODELS_PATH / MODELS_FILENAME,
            )

        m.end_run(test_mse)
    m.save(f"results/{run}")


if __name__ == "__main__":

    # main()
    explore_model_params()

    """
    Paramters for Graphino 1st model
    {
    "params": {
        "horizon": -1,
        "window": 3,
        "lon_min": 0,
        "lon_max": 360,
        "lat_min": -55,
        "lat_max": 60,
        "model_dir": "out/graphino/",
        "data_dir": "Data/",
        "useCMIP5": true,
        "use_heat_content": true,
        "seed": 41,
        "shuffle": true,
        "epochs": 50,
        "batch_size": 64,
        "lr": 0.005,
        "nesterov": true,
        "weight_decay": 1e-6,
        "validation_frac": 0,
        "validation_set": "SODA",
        "loss": "MSE",
        "optimizer": "SGD",
        "scheduler": "No"
    },

    "net_params": {
        "L": 2,
        "num_nodes": 1345,
        "readout": "mean",
        "activation": "ELU",
        "avg_edges_per_node": 8,
        "in_dim": 6,
        "adj_dim": 50,
        "jumping_knowledge": true,
        "index_node": true,
        "batch_norm": true,
        "mlp_batch_norm": true,
        "residual": true,
        "self_loop": true,
        "tanh_alpha": 0.1,
        "sig_alpha": 2.0,
        "dropout": 0.0,
        "hidden_dim": 250,
        "out_dim": 100
    }
}
"""
