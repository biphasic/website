import pytorch_lightning as pl
import sinabs
import sinabs.activation as sina
import sinabs.layers as sl
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import functional as F


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

    def __init__(self, lr: float, num_classes: int):
        super().__init__()
        self.save_hyperparameters()
        self.model = GestureClassifier(num_classes=num_classes)
        self.accuracy_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
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
        prediction = y_hat.argmax(1)
        accuracy = self.accuracy_metric(prediction, y)
        self.log("accuracy/valid", accuracy, prog_bar=True)
        self.log("loss/valid", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        prediction = y_hat.argmax(1)
        accuracy = self.accuracy_metric(prediction, y)
        self.log("accuracy/test", accuracy, prog_bar=True)
        self.log("hp_metric", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class SNN(pl.LightningModule):
    def __init__(self, lr: float, batch_size: int, num_classes: int):
        super().__init__()
        self.save_hyperparameters()
        cnn = GestureClassifier(num_classes=num_classes)
        self.accuracy_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.flatten_time = sl.FlattenTime()
        self.model = sinabs.from_torch.from_model(
            cnn,
            batch_size=batch_size,
            spike_fn=sina.MultiSpike,
            surrogate_grad_fn=sina.SingleExponential(),
            backend="exodus",
        ).spiking_model
        self.unflatten_time = sl.UnflattenTime(batch_size=batch_size)

    def forward(self, x):
        sinabs.reset_states(self.model)
        x = self.flatten_time(x)
        x = self.model(x)
        return self.unflatten_time(x)

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
        prediction = y_hat.argmax(1)
        accuracy = (prediction == y.long()).float().sum() / len(prediction)
        self.log("accuracy/valid", accuracy, prog_bar=True)
        self.log("loss/valid", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
