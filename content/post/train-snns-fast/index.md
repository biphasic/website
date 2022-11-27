---
title: "Training spiking neural networks, fast."

date: 2022-11-27
lastmod: 2022-11-27
draft: false

tags: ["SNN"]
summary: "How to use caching and EXODUS to speed up training by a factor of 10."
---

When training a spiking neural network (SNN), one might think about how the learning rate or model size affect training time. But when it comes to training *faster*, optimizing data movement is crucial. 3 out of the first 4 points in [this list](https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/) weighted after potential speed-up have to do with how data is shaped and moved around between actual computations. It makes a huge difference, because training faster means getting results faster!

For this post we train an SNN on the [Heidelberg Spiking Speech Commands](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) dataset to do audio stream classification. We'll benchmark different data loading strategies using [Tonic](https://github.com/neuromorphs/tonic) and show that with the right strategy, we can achieve a 10-fold speed-up compared to the naïve approach.

For all our benchmarks, we already assume multiple worker loading threads and pinning the GPU memory. We'll increase throughput by using different forms of caching to disk or GPU. By applying deterministic transformations upfront and saving the new tensor, we can save a lot of time during training. 
This tutorial is run on a machine with Ubuntu 20.04, an Intel Core i7-8700K CPU @ 3.70GHz, a Samsung SSD 850 and an NVIDIA GeForce RTX 2070 GPU.

All data from neuromorphic datasets in Tonic is provided as NxD numpy arrays. We'll need to transform this into a dense tensor to serve it to the GPU, and we'll also do some downsampling of time steps. Let's first define the transform. We know that samples of audio input data in this dataset are 0.8-1.2s long across 700 frequency channels at microsecond resolution. We'll [downsample](https://tonic.readthedocs.io/en/latest/reference/generated/tonic.transforms.Downsample.html#tonic.transforms.Downsample) each sample to 100 channels, [bin](https://tonic.readthedocs.io/en/latest/reference/generated/tonic.transforms.ToFrame.html#tonic.transforms.ToFrame) events every 4 ms to one frame and [cut](https://tonic.readthedocs.io/en/latest/reference/generated/tonic.transforms.CropTime.html#tonic.transforms.CropTime) samples that are longer than 1s. That leaves us with a maximum of 250 time steps per sample.


```python
from tonic import transforms

dt = 4000  # all time units in Tonic in us
encoding_dim = 100

dense_transform = transforms.Compose(
    [
        transforms.Downsample(spatial_factor=encoding_dim / 700),
        transforms.CropTime(max=1e6),
        transforms.ToFrame(
            sensor_size=(encoding_dim, 1, 1), time_window=dt, include_incomplete=True
        ),
    ]
)
```

Next we load the training dataset and assign the transform.


```python
from tonic import datasets

dense_dataset = datasets.SSC("./data", split="train", transform=dense_transform)
```

Let's plot one such dense tensor sample:


    
![png](index_files/index_6_0.png)
    


Next we define a spiking model. We use a simple integrate-and-fire (IAF) feed-forward architecture. For each dataloading method, we're going to test two different models. One is a [Sinabs](https://sinabs.readthedocs.io) model which is pretty much pure PyTorch plus for loops and the second one is an [EXODUS](https://github.com/synsense/sinabs-exodus) model, which is also based on PyTorch but vectorizes gradient computation for the time dimension using custom CUDA code. Both models compute the same activations and gradients, but the latter provides a significant speedup.


```python
import torch.nn as nn
import sinabs.layers as sl
import sinabs.exodus.layers as el


class SNN(nn.Sequential):
    def __init__(self, backend, hidden_dim: int = 128):
        assert backend == sl or backend == el
        super().__init__(
            nn.Linear(encoding_dim, hidden_dim),
            backend.IAF(),
            nn.Linear(hidden_dim, hidden_dim),
            backend.IAF(),
            nn.Linear(hidden_dim, hidden_dim),
            backend.IAF(),
            nn.Linear(hidden_dim, 35),
        )


sinabs_model = SNN(backend=sl).cuda()
exodus_model = SNN(backend=el).cuda()
```

## 1. Naïve dataloading

For the first benchmark we load every sample from an hdf5 file on disk which provides us with a numpy array in memory. For each sample, we apply our `dense_transform` defined earlier to create a dense tensor which we can then batch together with other samples and feed it to the network.

<figure>
  <img
  src="images/caching1.png"
  alt="Naïve caching">
  <figcaption>Figure 1: For every sample, we apply our transform ToFrame. The speed depends a lot on the CPU and the amount of worker threads used.</figcaption>
</figure>


```python
import sinabs
import timeit
import tonic
import pandas as pd
from torch.utils.data import DataLoader


dataloader_kwargs = dict(
    batch_size=128,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    collate_fn=tonic.collation.PadTensors(batch_first=True),
    num_workers=4,
)

naive_dataloader = DataLoader(dense_dataset, **dataloader_kwargs)


def training_loop(dataloader, model):
    for data, targets in iter(dataloader):
        data, targets = data.squeeze().cuda(), targets.cuda()
        sinabs.reset_states(model)
        output = model(data)
        loss = nn.functional.cross_entropy(output.sum(1), targets)
        loss.backward()
```


```python
from timeit import timeit

df.iloc[0, 0] = timeit(lambda: training_loop(naive_dataloader, sinabs_model), number=1)
df.iloc[3, 0] = timeit(lambda: training_loop(naive_dataloader, exodus_model), number=1)
```

{{< chart data="result1" >}}

The Sinabs model takes almost 200s per epoch using the simple strategy, which is far from exciting. By contrast, we can already see the huge speedup that EXODUS provides, almost halving the epoch time! These results are our baseline with the basic dataloading.

## Disk caching
Let's try to be a bit smarter now. ToFrame is a deterministic transform, so for the same sample we'll always receive the same transformed data. Given that we might train for 100 epochs, which looks at each sample 100 times, that's a lot of wasted compute! Now we're going to cache, which means save, those transformed samples to disk during the first epoch, so that we don't need to recompute them later on! To do this we simply wrap our previous dataset in a [DiskCachedDataset](https://tonic.readthedocs.io/en/latest/reference/data_classes.html#tonic.DiskCachedDataset) and provide the cache path. When a new sample is about to be loaded, that class will first check if the transformed sample is already in the cache on disk and if it isn't, it will retrieve the original sample, apply the transform, cache it to disk and then serve it. This caching process slows down training in the first epoch, but it pays off afterwards!

<figure>
  <img
  src="images/caching2.png"
  alt="Disk caching">
  <figcaption>Figure 2: During the first epoch, samples are transformed and then cached to disk. Afterwards, the transformed sample is loaded from disk straight away.</figcaption>
</figure>


```python
disk_cached_dataset = tonic.DiskCachedDataset(
    dataset=dense_dataset,
    cache_path=f"cache/{dense_dataset.__class__.__name__}/train/{encoding_dim}/{dt}",
)

disk_cached_dataloader = DataLoader(disk_cached_dataset, **dataloader_kwargs)
```


```python
# cache on disk already available
df.iloc[1, 0] = timeit(
    lambda: training_loop(disk_cached_dataloader, sinabs_model), number=1
)
df.iloc[4, 0] = timeit(
    lambda: training_loop(disk_cached_dataloader, exodus_model), number=1
)
```

{{< chart data="result2" >}}

We brought down epoch training time to 20s for the EXODUS model! The speedup comes at the expense of disk space. How much disk space does it cost you may ask? The size of the original dataset file is 2.65 GB compared to the generated cache folder of 1.04 GB, which is not too bad!

The original dataset contained numpy events, whereas the cache folder contains dense tensors. We can compress the dense tensors that much because by default Tonic uses lightweight compression during caching. So overall, disk-caching is generally applicable when training SNNs because it saves you the time to transform your events to dense tensors. Of course you could apply any other deterministic transform before caching it, and also easily apply augmentations to the cached samples as described in [this tutorial](https://tonic.readthedocs.io/en/latest/tutorials/fast_dataloading.html)!

Now we notice one more thing. Overall GPU utilisation rate at this point is at ~80%, which means that the GPU is still idling the rest of the time, waiting for new data to arrive. So we can try to go even faster!

## GPU caching
Instead of loading dense tensors from disk, we can try to cram all our dataset onto the GPU! Now, the issue is that with dense tensors this wouldn't work as they would occupy too much memory. But events are already an efficient format right? So we'll store the events on the GPU as sparse tensors and then simply inflate them as needed by calling to_dense() for each sample. This method is obviously bound by GPU memory so works with rather small datasets such as the one we're testing. However, once you're setup, you can train with _blazing_ speed. For GPU caching we are going to:

1. Create a new sparse dataset by loading them from the disk cache and calling to_sparse() on the transformed tensors.
2. Create a new dataloader that now uses a single thread.
3. Inflate sparse tensors to dense versions by calling to_dense() in the training loop.

<figure>
  <img
  src="images/caching3.png"
  alt="Disk caching">
  <figcaption>Figure 3: During the first epoch, transformed samples are loaded onto the GPU and stored in a list of sparse tensors. Whenever a new sample is needed, is is inflated by to_dense() and fed to the network. This process is almost instantaneous and now bound by what your model can process.</figcaption>
</figure>

The sparse tensor dataset takes about 5.7 GB of GPU memory. Not exactly efficient, but also not terrible. What about training speeds?


```python
def gpu_training_loop(model):
    for data, targets in iter(sparse_tensor_dataloader):
        data = data.to_dense()
        sinabs.reset_states(model)
        output = model(data)
        loss = nn.functional.cross_entropy(output.sum(1), targets)
        loss.backward()
```


```python
df.iloc[2, 0] = timeit(lambda: gpu_training_loop(sinabs_model), number=1)
df.iloc[5, 0] = timeit(lambda: gpu_training_loop(exodus_model), number=1)
```

{{< chart data="result3" >}}

We're down to 16s per epoch for the EXODUS model, which is a ten-fold improvement over the original Sinabs model using the naïve dataloading approach! All this without any impact whatsoever on training performance. 

## Conclusion
By using cached samples and not having to recompute the same transformations every time, we save ourselves a lot of time during training. If the data already sits on the GPU when it is requested, the speedup is really high. After all, there is a reason why neural network accelerators heavily optimise caching and reuse of data to minimise time and energy spent on data movement. So when should you use either disk- or GPU-caching?

* Disk-caching: Broadly applicable, useful if you apply deterministic transformations to each sample and you train for many epochs. Not ideal if you're low on disk space.
* GPU-caching: Only really suitable for small datasets and a bit more intricate to setup, but well worth the effort if you want to explore many different architectures / training parameters due to the speed of iteration.

As a last note, you might be wondering why we don't cache to the host memory instead of reading from a disk cache. This is totally possible, but the bottleneck at that point really is moving the data onto the GPU, which takes time. Whether the data sits in host memory or is loaded from disk using multiple worker threads doesn't make much of a difference, because the GPU cannot handle the data movement. Since on disk we have much more space available than in RAM, we normally choose to do that.

Acknowledgements: Thanks a lot to Omar Oubari for the nice feedback, as always.
