import pytorch_lightning as pl
import sinabs
import sinabs.activation as sina
import sinabs.layers as sl
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import functional as F
from torchmetrics.classification import MulticlassConfusionMatrix
import tonic
from mlxtend.plotting import plot_confusion_matrix


class GestureClassifier(nn.Sequential):
    def __init__(self, num_classes: int):
        super().__init__(
            nn.Conv2d(2, 16, kernel_size=2, stride=2, bias=False),
            nn.ReLU(),
            # core 1
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # core 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # core 7
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # core 4
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # core 5
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            # nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2),
            # core 6
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 256, kernel_size=2, bias=False),
            nn.ReLU(),
            # core 3
            nn.Dropout2d(0.5),
            nn.Flatten(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            # core 8
            nn.Linear(128, num_classes, bias=False),
        )


class CNN(pl.LightningModule):
    """A simple CNN which uses the GestureClassifier as a backend.

    Parameters:
        lr: The learning rate.
        num_classes: The number of output neurons / classes.
    """

    def __init__(self, num_classes: int, batch_size: int):
        super().__init__()
        self.save_hyperparameters()
        self.model = GestureClassifier(num_classes=num_classes)
        self.valid_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )
        self.conf_matrix = MulticlassConfusionMatrix(
            normalize="true", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("loss/valid", loss)
        self.valid_accuracy(y_hat, y)
        self.log("accuracy/valid", self.valid_accuracy, prog_bar=True)
        self.conf_matrix.update(y_hat, y)

    def on_validation_epoch_end(self) -> None:
        matrix = self.conf_matrix.compute()
        figure, _ = plot_confusion_matrix(
            conf_mat=matrix.cpu().numpy(),
            class_names=tonic.datasets.DVSGesture.classes,
            show_absolute=False,
            show_normed=True,
            colorbar=True,
            figsize=(6, 9),
        )
        figure.tight_layout()
        self.logger.experiment.add_figure(
            "confusion matrix", figure=figure, global_step=self.global_step
        )
        self.conf_matrix.reset()
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.test_accuracy(y_hat, y)
        self.log("accuracy/test", self.test_accuracy, prog_bar=True)
        self.log("hp_metric", self.test_accuracy)


class SNN(pl.LightningModule):
    def __init__(self, lr: float, batch_size: int, num_classes: int):
        super().__init__()
        self.save_hyperparameters()
        cnn = GestureClassifier(num_classes=num_classes)
        self.valid_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )
        self.flatten_time = sl.FlattenTime()
        self.model = sinabs.from_torch.from_model(
            cnn,
            batch_size=batch_size,
            spike_fn=sina.MultiSpike,
            surrogate_grad_fn=sina.SingleExponential(),
            spike_layer_class=sinabs.exodus.layers.IAF,
            min_v_mem=None,
        ).spiking_model
        self.unflatten_time = sl.UnflattenTime(batch_size=batch_size)

    def forward(self, x):
        sinabs.reset_states(self.model)
        batch_size, num_timesteps = x.shape[:2]
        x = self.model(x.flatten(0, 1))
        return x.unflatten(0, (batch_size, num_timesteps))
        # x = self.flatten_time(x)
        # x = self.model(x)
        # return self.unflatten_time(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).sum(1)
        loss = F.cross_entropy(y_hat, y)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).sum(1)
        loss = F.cross_entropy(y_hat, y)
        self.log("loss/valid", loss)
        self.valid_accuracy(y_hat, y)
        self.log("accuracy/valid", self.valid_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).sum(1)
        self.test_accuracy(y_hat, y)
        self.log("accuracy/test", self.test_accuracy, prog_bar=True)
        self.log("hp_metric", self.test_accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
