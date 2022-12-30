import hashlib
import os
import torch
from typing import Callable, Iterable, Optional

import numpy as np
import pytorch_lightning as pl
from tonic import (
    DiskCachedDataset,
    MemoryCachedDataset,
    SlicedDataset,
    datasets,
    transforms,
)
from torch.utils.data import DataLoader
import tonic


class DVSGesture(pl.LightningDataModule):
    """
    Parameters:
        batch_size: How many images to bundle up in one mini batch.
        data_dir: Path that points to the dataset folder.
        preprocess: An optional list of callables that will be applied to each sample before slicing the dataset.
        preprocess2: An optional list of callables that will be applied to each sliced sample, before caching.
        augmentation: An optional list of callables that will be applied to each cached sample before feeding it to the model.
        slicer: An optional Tonic slicer callable that decides how to cut one recording into smaller samples, for example based on time or a number of events.
        cache_path: Where to store cached versions of all the frames.
        metadata_path: Store metadata about how recordings are sliced in individual samples.
                       Providing the path to store the metadata saves time when loading the dataset the next time.
        num_workers: The number of threads for the dataloader.
    """

    def __init__(
        self,
        batch_size: int,
        data_dir: str = "data",
        preprocess: Optional[Iterable[Callable]] = None,
        preprocess2: Optional[Iterable[Callable]] = None,
        augmentation: Optional[Iterable[Callable]] = None,
        slicer: Optional[tonic.slicers.Slicer] = None,
        cache_path: str = "cache",
        metadata_path: str = "metadata",
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        """Download and unpack data if not already available."""
        datasets.DVSGesture(
            save_to=self.hparams.data_dir,
            train=True,
        )
        datasets.DVSGesture(
            save_to=self.hparams.data_dir,
            train=False,
        )

    def get_train_or_testset(self, train: bool):
        dataset = datasets.DVSGesture(
            save_to=self.hparams.data_dir,
            train=train,
            transform=transforms.Compose(self.hparams.preprocess),
        )

        dataset = MemoryCachedDataset(dataset=dataset)

        to_float = lambda x: x.astype(np.float32)
        dataset = SlicedDataset(
            dataset=dataset,
            slicer=self.hparams.slicer,
            metadata_path=os.path.join(
                self.hparams.metadata_path,
                "train" if train else "test",
                hashlib.md5(
                    (str(self.hparams.preprocess) + str(self.hparams.slicer)).encode(
                        "utf-8"
                    )
                ).hexdigest(),
            ),
            transform=transforms.Compose(self.hparams.preprocess2 + [to_float]),
        )

        to_tensor = lambda x: torch.as_tensor(x)
        return DiskCachedDataset(
            dataset=dataset,
            cache_path=os.path.join(
                self.hparams.cache_path,
                "train" if train else "test",
                hashlib.md5(
                    (
                        str(self.hparams.preprocess + self.hparams.preprocess2)
                        + str(self.hparams.slicer)
                    ).encode("utf-8")
                ).hexdigest(),
            ),
            transform=transforms.Compose([to_tensor] + self.hparams.augmentation),
        )

    def setup(self, stage=None):
        self.train_data = self.get_train_or_testset(train=True)
        self.valid_data = self.get_train_or_testset(train=False)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            prefetch_factor=4,
        )

    def test_dataloader(self):
        return self.val_dataloader()
