import re
import os
import time
import torch
import pathlib
import numpy as np
import networkx as nx

from itertools import product
from collections import namedtuple
from datetime import datetime as dt
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from src.config import RUNS_PATH, MODELS_PATH


class RunBuilder:
    @staticmethod
    def get_runs(params):
        # Create a new subclass of Tuple with named fields
        Run = namedtuple("Run", params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


class RunManager:
    def __init__(self, val=False, verbose=True):
        """
        val: if we're using a validation set, RunManager will log the validation metrics
        verbose: if you want to print interim results for each epoch
        """
        self.val = val
        self.verbose = verbose
        self.epoch_r2 = []
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_start_time = None
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.model = None
        self.loader = None
        self.tb = None

        pathlib.Path(RUNS_PATH).mkdir(parents=True, exist_ok=True)
        self.previous_runs = os.listdir(RUNS_PATH)

        pathlib.Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)

    def check_run_already_done(self, run) -> bool:

        run_str = re.escape(run.__str__())
        r = re.compile(rf".*{run_str}")

        matches = list(filter(r.match, self.previous_runs))

        return len(matches) > 0

    def begin_run(self, run, model, loader):
        self.run = run
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1

        self.model = model
        self.model.train()
        self.loader = loader
        run_date = dt.today().strftime("%Y-%m-%d")
        self.tb = SummaryWriter(log_dir=f"runs/{run_date}-{run}")

        print(run)

    def end_run(self, test_mse, test_r2, save_model=False):
        run_duration = time.time() - self.run_start_time

        A = self.model.A.detach().numpy()
        A = nx.from_numpy_array(A)
        num_edges = A.number_of_edges()

        self.tb.add_histogram("Run Duration", run_duration)
        self.tb.add_scalar("Testset MSE", test_mse)
        self.tb.flush()
        self.tb.close()

        print(f"Test Set:\n\tMSE {test_mse}\n\tR2 {test_r2}\n\tEdges {num_edges}")

        if save_model:
            if test_mse < 1 and test_r2 < 1:
                print(
                    f"Test MSE {test_mse} < 1 and R2 {test_r2} < 100. Saving model and Adjacency matrix"
                )
                MODEL_FILENAME = (
                    f"GCN-{self.epoch_count}ep-MSE={test_mse}-TestR2={test_r2}-{self.run}.pth"
                )
                torch.save(
                    self.model,
                    MODELS_PATH / MODEL_FILENAME,
                )

        with open("all_runs.txt", "a") as out:
            out.write(
                f"{self.run}\n\tMSE:\t{test_mse}\n\tR2: \t{test_r2}\n\tEdges: {num_edges}\n\n"
            )

        self.epoch_count = 0

    def begin_epoch(self) -> None:
        torch.enable_grad()
        self.model.train()
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_r2 = []

    def end_epoch(self, val_mse=0, val_r2=0):

        epoch_duration = time.time() - self.epoch_start_time

        A = self.model.A.detach().numpy()
        A = nx.from_numpy_array(A)
        num_edges = A.number_of_edges()

        total_loss = self.epoch_loss / len(self.loader.dataset)
        epoch_r2 = np.mean(self.epoch_r2)

        self.tb.add_scalar("Loss (MSE)", total_loss, self.epoch_count)
        self.tb.add_scalar("Epoch R2", epoch_r2, self.epoch_count)
        self.tb.add_scalar("Num Edges", num_edges, self.epoch_count)

        if self.val:
            self.tb.add_scalar("Val Loss (MSE)", val_mse, self.epoch_count)
            self.tb.add_scalar("Val R2", val_r2, self.epoch_count)

        self.tb.add_histogram("Epoch Duration", epoch_duration, self.epoch_count)

        for name, weight in self.model.named_parameters():
            try:
                self.tb.add_histogram(name, weight, self.epoch_count)
                self.tb.add_histogram(f"{name}.grad", weight.grad, self.epoch_count)
            except ValueError:
                print("Caught empty model param")
                continue

        if self.verbose:
            print(
                "---------------------------------------------------"
                f"\t\nEpoch {self.epoch_count}:\n\tEpoch MSE:\t{total_loss}\n\t"
                f"Epoch R2:\t{epoch_r2}\n\tNum edges:\t{num_edges}"
            )
            if self.val:
                print(f"\n\tValidation MSE:\t{val_mse}\n\tValidation R2:\t{val_r2}")

    def track_loss(self, loss):
        # MSE * batch_size gives SSE
        # Don't forget to divide by num_smaples at the end
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_r2(self, r2):
        self.epoch_r2.append(r2)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()


if __name__ == "__main__":

    params = OrderedDict(
        lr=[0.01, 0.001, 0.0001], batch_size=[10, 100, 1000], shuffle=[True, False]
    )

    runs = RunBuilder.get_runs(params)

    print(runs)
