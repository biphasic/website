import pytorch_lightning as pl
import sinabs
import sinabs.activation as sina
import tonic
import torch.nn as nn
import torchmetrics
from mlxtend.plotting import plot_confusion_matrix
from torch.nn import functional as F
from torchmetrics.classification import MulticlassConfusionMatrix

try:
    from sinabs.exodus.layers import IAFSqueeze, LIFSqueeze
except ImportError:
    print("Exodus not available.")
    from sinabs.layers import IAFSqueeze, LIFSqueeze


class GestureClassifier(nn.Sequential):
    def __init__(self, num_classes: int):
        bias = True
        dropout_p=0.3
        base_channel = 2
        super().__init__(
            nn.Conv2d(2, base_channel, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_channel, base_channel*2, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_channel*2, base_channel*4, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_channel*4, base_channel*8, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256, num_classes, bias=bias),
        )


class CNN(pl.LightningModule):
    """A simple CNN which uses the GestureClassifier as a backend.

    Parameters:
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
            num_classes=num_classes,  # normalize="true",
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
            figsize=(8, 6),
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
    def __init__(self, num_classes: int, batch_size: int):
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
        self.conf_matrix = MulticlassConfusionMatrix(
            num_classes=num_classes,
        )
        self.model = sinabs.from_torch.from_model(
            cnn,
            batch_size=batch_size,
            spike_fn=sina.SingleSpike,
            surrogate_grad_fn=sina.SingleExponential(),
            spike_layer_class=LIFSqueeze,
            min_v_mem=None,
            spike_threshold=0.25,
            kwargs_backend={'tau_mem':10.}
        ).spiking_model

    def forward(self, x):
        sinabs.reset_states(self.model)
        batch_size, num_timesteps = x.shape[:2]
        x = self.model(x.flatten(0, 1))
        return x.unflatten(0, (batch_size, num_timesteps))

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
        # import ipdb; ipdb.set_trace()
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
            figsize=(8, 6),
        )
        figure.tight_layout()
        self.logger.experiment.add_figure(
            "confusion matrix", figure=figure, global_step=self.global_step
        )
        self.conf_matrix.reset()
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).sum(1)
        self.test_accuracy(y_hat, y)
        self.log("accuracy/test", self.test_accuracy, prog_bar=True)
        self.log("hp_metric", self.test_accuracy)
