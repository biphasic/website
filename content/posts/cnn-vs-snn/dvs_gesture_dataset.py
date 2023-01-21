import hashlib
import os
from typing import Callable, Iterable, Optional

import pytorch_lightning as pl
import tonic
from tonic import (
    DiskCachedDataset,
    MemoryCachedDataset,
    SlicedDataset,
    datasets,
    transforms,
)
from torch.utils.data import DataLoader


class DVSGesture(pl.LightningDataModule):
    """
    Parameters:
        batch_size: How many images to bundle up in one mini batch.
        data_dir: Path that points to the dataset folder.
        pre_slicing_transform: An optional list of callables that will be applied to each sample before slicing the dataset.
        post_slicing_transform: An optional list of callables that will be applied to each sliced sample, before caching.
        augmentation: An optional list of callables that will be applied to each cached sample before feeding it to the model.
        slicer: An optional Tonic slicer callable that decides how to cut one recording into smaller samples, for example based on time or a number of events.
        cache_path: Where to store cached versions of all the frames.
        metadata_path: Store metadata about how recordings are sliced in individual samples.
                       Providing the path to store the metadata saves time when loading the dataset the next time.
        num_workers: The number of threads for the dataloader.
        prefetch_factor: The prefetch_factor arg of the dataloader, numbers of batches loaded in advance by each worker.
        data_loader_collate_fn: The collate_fn arg of the dataloader, which defines how to group a batch of data together
    """

    def __init__(
        self,
        batch_size: int,
        data_dir: str = "data",
        pre_slicing_transform: Optional[Iterable[Callable]] = None,
        post_slicing_transform: Optional[Iterable[Callable]] = None,
        augmentation: Optional[Iterable[Callable]] = None,
        slicer: Optional[tonic.slicers.Slicer] = None,
        cache_path: str = "cache",
        metadata_path: str = "metadata",
        num_workers: int = 4,
        prefetch_factor: int = 4,
        data_loader_collate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        print(
            f"Your deterministic transforms are as follows:\nPre-slicing\n\t{pre_slicing_transform}"
            + f"\nSlicer\n\t{slicer}\nPost-Slicing\n\t{post_slicing_transform}\n"
            + f"If you changed a parameter and it doesn't show up here, the cache path will not change!"
        )

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
        compose = lambda x: transforms.Compose(x) if x is not None else None
        dataset = datasets.DVSGesture(
            save_to=self.hparams.data_dir,
            train=train,
            transform=compose(self.hparams.pre_slicing_transform),
        )

        dataset = MemoryCachedDataset(dataset=dataset)

        hash_fn = lambda x: hashlib.md5((x).encode("utf-8")).hexdigest()
        if self.hparams.slicer is not None:
            dataset = SlicedDataset(
                dataset=dataset,
                slicer=self.hparams.slicer,
                metadata_path=os.path.join(
                    self.hparams.metadata_path,
                    "train" if train else "test",
                    hash_fn(
                        str(self.hparams.pre_slicing_transform)
                        + str(self.hparams.slicer)
                    ),
                ),
                transform=compose(self.hparams.post_slicing_transform),
            )

        return DiskCachedDataset(
            dataset=dataset,
            cache_path=os.path.join(
                self.hparams.cache_path,
                "train" if train else "test",
                hash_fn(
                    str(self.hparams.pre_slicing_transform)
                    + str(self.hparams.post_slicing_transform)
                    + str(self.hparams.slicer)
                ),
            ),
            transform=compose(self.hparams.augmentation) if train else None,
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
            prefetch_factor=self.hparams.prefetch_factor,
            drop_last=True,
            collate_fn=self.hparams.data_loader_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            prefetch_factor=self.hparams.prefetch_factor,
            drop_last=True,
            collate_fn=self.hparams.data_loader_collate_fn,
        )

    def test_dataloader(self):
        return self.val_dataloader()
