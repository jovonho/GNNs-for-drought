import time
import torch
import numpy as np

from itertools import product
from collections import OrderedDict
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self):
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

    def begin_run(self, run, model, loader):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1

        self.model = model
        self.model.train()
        self.loader = loader
        self.tb = SummaryWriter(comment=f"-{run}")

        print(run)

    def end_run(self, test_mse):
        run_duration = time.time() - self.run_start_time
        self.tb.add_histogram("Run Duration", run_duration)
        self.tb.add_scalar("Testset MSE", test_mse)
        self.tb.flush()
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_r2 = []

    def end_epoch(self, num_edges, val_mse, val_r2):

        epoch_duration = time.time() - self.epoch_start_time

        total_loss = self.epoch_loss / len(self.loader.dataset)
        epoch_r2 = np.mean(self.epoch_r2)

        self.tb.add_scalar("Loss (MSE)", total_loss, self.epoch_count)
        self.tb.add_scalar("Val Loss (MSE)", val_mse, self.epoch_count)
        self.tb.add_scalar("Epoch R2", epoch_r2, self.epoch_count)
        self.tb.add_scalar("Val R2", val_r2, self.epoch_count)
        self.tb.add_scalar("Num Edges", num_edges, self.epoch_count)

        self.tb.add_histogram("Epoch Duration", epoch_duration, self.epoch_count)

        for name, weight in self.model.named_parameters():
            try:
                self.tb.add_histogram(name, weight, self.epoch_count)
                self.tb.add_histogram(f"{name}.grad", weight.grad, self.epoch_count)
            except ValueError:
                print("Caught empty model param")
                continue

        print(
            "---------------------------------------------------"
            f"\t\nEpoch {self.epoch_count}:\n\tEpoch MSE:\t{total_loss}\n\t"
            f"Epoch R2:\t{epoch_r2}\n\tNum edges:\t{num_edges}\n\t"
            f"Validation MSE:\t{val_mse}\n\tValidation R2:\t{val_r2}"
        )

    def track_loss(self, loss):
        # MSE * batch_size gives SSE
        # Don't forget to divide by num_smaples at the end
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_r2(self, r2):
        self.epoch_r2.append(r2)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    # def save(self, filename):
    #     df = pd.DataFrame.from_dict(self.run_data, orient="columns").sort_values(
    #         "R2", ascending=False
    #     )
    #     print(df)
    #     df.to_csv(f"{filename}.csv")

    #     with open(f"{filename}.json", "w", encoding="utf-8") as f:
    #         json.dump(self.run_data, f, indent=4)


if __name__ == "__main__":

    params = OrderedDict(
        lr=[0.01, 0.001, 0.0001], batch_size=[10, 100, 1000], shuffle=[True, False]
    )

    runs = RunBuilder.get_runs(params)

    print(runs)