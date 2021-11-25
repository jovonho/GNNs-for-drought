import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from pathlib import Path
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data import Dataset, DATAFOLDER_PATH
from ..utils import filter_preds_test_by_mask
from sklearn.metrics import mean_squared_error

from typing import List


class TrainValDataset(Dataset):
    def __init__(
        self,
        data_folder: Path = DATAFOLDER_PATH,
        is_val: bool = False,
        val_ratio: float = 0.2,
        flatten: bool = True,
    ) -> None:
        super().__init__(data_folder, is_test=False, flatten=flatten)

        differentiator_index = int(len(self.time_pairs) * (1 - val_ratio))
        if is_val:
            self.time_pairs = self.time_pairs[differentiator_index:]
        else:
            self.time_pairs = self.time_pairs[:differentiator_index]


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_vector_sizes: List,
        batch_norm: bool = False,
        dropout_val: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        hidden_vector_sizes.insert(0, in_features)
        hidden_vector_sizes.append(out_features)

        model_layers = []
        for idx in range(len(hidden_vector_sizes) - 2):
            model_layers.extend(
                [
                    nn.Linear(hidden_vector_sizes[idx], hidden_vector_sizes[idx + 1]),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_val),
                    nn.BatchNorm1d(hidden_vector_sizes[idx + 1]),
                ]
            )
        # Append the last layer without dropout or batchnorm
        idx = len(hidden_vector_sizes) - 2
        model_layers.extend(
            [
                nn.Linear(hidden_vector_sizes[idx], hidden_vector_sizes[idx + 1]),
                nn.ReLU(),
            ]
        )
        self.layers = nn.Sequential(*model_layers)

    def forward(self, x):
        return self.layers(x)


class MLPTrainer(pl.LightningModule):
    def __init__(
        self,
        root,
        batch_size: int,
        learning_rate: float,
        val_ratio: float,
        model_name: str,
    ) -> None:
        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.val_ratio = val_ratio

        (Path(self.root) / self.model_name).mkdir(exist_ok=True)

        dataset = self.get_dataset(mode="train")

        self.classifier = MLP(
            in_features=dataset.num_features,
            out_features=dataset.num_predictands,
            hidden_vector_sizes=[512, 512],
        )

        self.best_val_loss: float = np.inf

        self.loss = nn.SmoothL1Loss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)

    def get_dataset(self, mode: str) -> Dataset:
        if mode == "test":
            return Dataset(DATAFOLDER_PATH, is_test=True)
        else:
            return TrainValDataset(
                DATAFOLDER_PATH, is_val=True if mode == "val" else False, val_ratio=self.val_ratio
            )

    def train_dataloader(self):
        return DataLoader(self.get_dataset(mode="train"), shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.get_dataset(mode="val"), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.get_dataset(mode="test"), batch_size=self.batch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch

        preds = self.forward(x.float())
        y, preds = filter_preds_test_by_mask(y, preds.squeeze(1), mask)
        loss = self.loss(preds, y.float())

        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch

        preds = self.forward(x.float())
        y, preds = filter_preds_test_by_mask(y, preds.squeeze(1), mask)
        loss = self.loss(preds, y.float())

        return {"val_loss": loss, "log": {"val_loss": loss}, "preds": preds, "labels": y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()

        if len(np.unique(labels)) == 1:
            # this happens during the sanity check
            return {
                "val_loss": avg_loss,
            }

        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_MSE": mean_squared_error(labels, preds),
        }

        if float(avg_loss) < self.best_val_loss:
            self.best_val_loss = float(avg_loss)
            print(f"\n Saving best state_dict - MSE: {tensorboard_logs['val_MSE']}")
            self.save_state_dict()

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def save_state_dict(self) -> None:
        torch.save(
            self.classifier.state_dict(), Path(self.root) / self.model_name / "state_dict.pth"
        )


def train_mlp(
    root,
    model_name: str,
    val_ratio: float = 0.2,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    max_epochs: int = 1000,
    patience: int = 10,
) -> MLP:
    r"""
    Initialize and pretrain a classifier on a global crop vs. non crop task
    :root: The path to the data
    :param classifier_vector_size: The LSTM hidden vector size to use
    :param classifier_dropout: The value for variational dropout between LSTM timesteps to us
    :param classifier_base_layers: The number of LSTM layers to use
    :param num_classification_layers: The number of linear classification layers to use on top
        of the LSTM base
    :param model_name: The model name. The model's weights will be saved at root / model_name.
    :param pretrained_val_ratio: The ratio of data to use for validation (for early stopping)
    :param batch_size: The batch size to use when pretraining the model
    :param learning_rate: The learning rate to use
    :param max_epochs: The maximum number of epochs to train the model for
    :param patience: The patience to use for early stopping. If the model trains for
        `patience` epochs without improvement on the validation set, training ends
    """

    model = MLPTrainer(
        root,
        batch_size,
        learning_rate,
        val_ratio,
        model_name,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=patience, verbose=True, mode="min"
    )
    trainer = pl.Trainer(
        default_save_path=Path(model.root),
        max_epochs=max_epochs,
        early_stop_callback=early_stop_callback,
    )
    trainer.fit(model)

    return model.classifier
