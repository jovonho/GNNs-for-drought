import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from src.models.gcn import GCN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score, mean_squared_error

from src.data import Dataset

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
    print(model)

    log_iter = (num_samples // batch_size) // 4

    for epoch in range(epochs):

        total_loss = 0
        epoch_r2 = []
        progress_loss = 0
        progress_r2 = []

        for i, batch in enumerate(trainloader):
            X, y_true = batch

            y_pred = model(X)
            loss = criterion(y_pred, y_true)
            r2 = r2_score(y_true.data, y_pred.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Are we supposed to multiply these by the batch_size?
            total_loss += loss.item()
            progress_loss += loss.item()
            epoch_r2.append(r2)
            progress_r2.append(r2)

            if i % log_iter == 0 and i != 0:
                print(f"Iteration {i}:\n\tMSE:\t{progress_loss}\n\tr2:\t{np.mean(progress_r2)}")
                progress_loss = 0
                progress_r2 = []

        epoch_r2 = np.mean(epoch_r2)
        num_edges = len(torch.nonzero(model.A.detach(), as_tuple=False))
        print(
            "---------------------------------------------------"
            f"\t\nEpoch {epoch}:\n\ttotal MSE:\t{total_loss}\n\t"
            f"epoch r2:\t{epoch_r2}\n\tnum edges:\t{num_edges}\n"
        )

        tb.add_scalar("MSE", total_loss, epoch)
        tb.add_scalar("R2", epoch_r2, epoch)
        tb.add_scalar("Num Edges", num_edges, epoch)

        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f"{name}.grad", weight.grad, epoch)

    print(f"Running time: {time.time() - t1} s")
    tb.flush()
    tb.close()

    torch.save(
        model,
        Path(f"./models/GCN-{epochs}epochs-{lr}lr-{batch_size}bs-{hid_dim}hd-{num_layer}L.pth"),
    )

    # Evaluate on test set
    model.eval()
    mse = 0
    r2 = 0
    for i, (X, y_true) in enumerate(testloader):
        preds = model(X)
        r2 += r2_score(y_true.data, preds.data)
        mse += mean_squared_error(y_true.data, preds.data)

    r2 = np.mean(r2)
    print(f"Test Set:\n\tMSE {mse}\n\tR2 {r2}")


if __name__ == "__main__":
    main()

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
