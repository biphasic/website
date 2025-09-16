---
title: "Event cameras in 2025, Part 2"

commentable: true

date: 2025-08-20
lastmod: 2025-08-20
draft: false

tags: ["Event cameras"]
summary: "Technological challenges that are to be overcome before event cameras enter the mass market."
---

In [Part 1](https://lenzgregor.com/posts/event-cameras-2025-part1/) I provided a high level overview of different industry sectors that could potentially see the adoption of event cameras. Apart from the challenge of finding the right application, there are several technological challenges before event cameras can reach a mass audience. 

## Sensor Capabilities
Today's most recent event cameras are summarised in the table below.

| Camera Supplier | Sensor | Model Name | Year | Resolution | Dynamic Range (dB) | Max Bandwidth (Mev/s) |
|-----------------|--------|------------|------|------------|-----------|-----------------------|
| iniVation       | Gen2 DVS | [DAVIS346](https://docs.inivation.com/hardware/current-products/davis346.html) | 2017 | 346×260 | ~120 | 12 |
| iniVation       | Gen3 DVS | [DVXPlorer](https://docs.inivation.com/hardware/current-products/dvxplorer.html) | 2020 | 640×480 | 90-110 | 165  |
| Prophesee       | [Sony IMX636](https://www.prophesee.ai/event-based-sensor-imx636-sony-prophesee/) | [EVK4](https://www.prophesee.ai/event-camera-evk4/) | 2020 | 1280×720 | 120 | 1066 |
| Prophesee       |  [GenX320](https://www.prophesee.ai/event-based-sensor-genx320/) | [EVK3](https://www.prophesee.ai/evk-3-genx320-info/) | 2023 | 320×320 | 140 |  |
| Samsung         | Gen4 DVS | DVS-Gen4 | 2020 | 1280×960 |  | 1200 |

Insightness was sold to Sony, and CelePixel partnered with Omnivision, but hasn't released anything in the past 5 years. Over the past decade, we have seen resolution grow from 128x128 to HD, but that's actually not always good. The last column in the table above describes the number of million events per second, which can easily be reached when the camera is moving fast, such on a drone. A paper by [Gehrig and Scaramuzza](https://arxiv.org/abs/2203.14672) suggests that in low light and high speed scenarios, performance of high res cameras is actually worse than when using fewer, but bigger pixels, due to high per-pixel event rates that are noisy and cause ghosting artifacts.  

In areas such as defence, higher resolution and contrast sensitivity, as well as capturing the short/mid range infrared spectrum, is going to be desirable, because range is so important. SCD USA made the [MIRA 02Y-E](https://scdusa-ir.com/wp-content/uploads/2024/06/Mira_V1g.pdf) available last year that includes an optional event-based readout, to enable tactical forces to detect laser sources. Using the event-based output, it advertises a frame rate of up to 1.2 kHz. In space, the distances to the captured objects are enormous, and therefore high resolution and light sensitivity are of utmost importance.

In short range applications such as eye tracking for wearables, a [GenX320](https://www.prophesee.ai/event-based-sensor-genx320/) at lower resolution but high dynamic range and ultra low power modes is going to be more interesting. 
For scientific applications, NovoViz recently [announced](https://www.tokyoupdates.metro.tokyo.lg.jp/en/post-1551/) a new SPAD (single photon avalanche diode) camera using event-based outputs!

One thing is clear: today’s binary microsecond spikes are rarely the right format. Much like Intel’s [Loihi 2](https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-2-intel/) shifted from binary spikes to richer spike payloads because they realised that the communication overhead was too high otherwise, future event cameras could emit multi-bit “micro-frames” or tokenizable spike packets. These would represent short-term local activity and could be directly ingested by ML models, reducing the need for preprocessing altogether. Ideally there’s a trade-off between information density and temporal resolution that can be chosen depending on the application. 

A key trend are hybrid vision sensors that combine rgb and event frames. Early designs such as the DAVIS output RGB frames and raw events, which then needed to be converted to other representations. At ISSCC 2023, three papers ([one](https://ieeexplore.ieee.org/document/10067520), [two](https://ieeexplore.ieee.org/document/10067566), and [three](https://ieeexplore.ieee.org/document/10067476)) were released that showed *event frames* at variable rate being output next to fixed RGB outputs. While many researchers (including me) originally set out to discard event frames, I think it is time to swallow a bitter pill and accept that computer vision will depend on it for the foreseeable future. 

![hybrid-vision-sensors](images/hvs.png)
*[Kodama et al.](https://ieeexplore.ieee.org/document/10067520) presented a sensor that outputs variable event frame rates next to RGB.* 

In either case, the event camera sensor has not reached its final form yet. Binary events don't contain enough information on their own, so they must be aggregated in one form or another. Event sensors might move from binary outputs toward richer encodings at the pixel level, attach a dedicated processor to output richer representations, or they simply output what the world already knows well: another form of frames. My bet is currently on the latter, because the simplest solutions tend to win. 

## Event Representations
Most common approaches aggregate events into [image-like representations](https://tonic.readthedocs.io/en/latest/auto_examples/index.html#event-representations) such as 2d histograms, voxel grids, or time surfaces. These are then used to fine-tune deep learning models that were pre-trained on RGB images. This leverages the breadth of existing tooling built for images and is compatible with GPU-accelerated training and inference. Moreover, it allows for adaptive frame rates, aggregating only when there’s activity and potentially saving on compute. However, this method discards much of the fine temporal structure that makes event cameras valuable in the first place. 
We still lack a representation for event streams that works well with modern ML architectures and preserves their sparsity. Event streams are a new data modality, just like images, audio, or text, but one for which we haven’t yet cracked the “tokenization problem.” A single ON or OFF event contains very little semantic information. Unlike a word in a sentence, which can encode a concept, even a dozen events reveal almost nothing about the scene. This makes direct tokenization of events inefficient and ineffective. What we need is a representation that can summarize local spatiotemporal structure into meaningful, higher-level primitives. Something akin to a “visual word” for events.

It’s also inherently inefficient: the tensors produced are full of zeros, and latency grows with the size of the memory window. This becomes problematic for real-time applications where a long temporal context is needed but high responsiveness is crucial.


I think that graphs, especially dynamic, sparse graphs, are an interesting abstraction to be explored. Each node could represent a small region of correlated activity in space and time, with edges encoding temporal or spatial relationships. Recent work such as [HugNet v2](https://openaccess.thecvf.com/content/CVPR2025/html/Dampfhoffer_Graph_Neural_Network_Combining_Event_Stream_and_Periodic_Aggregation_for_CVPR_2025_paper.html), [DAGr](https://www.nature.com/articles/s41586-024-07409-w), or [EvGNN hardware](https://ieeexplore.ieee.org/abstract/document/10812004) apply Graph Neural Networks (GNNs) to event data. But several challenges remain: to generate such a graph, we need a lot of memory for all those events, and the upredictable number of incoming events makes computation extremely inefficient. This is where specialized hardware accelerators will need to come in, because dynamically fetching events is expensive. By combining event cameras with efficient “graph processors,” we could offload the task of building sparse graphs directly on-chip, producing representations that are ready for downstream learning. Temporally sparse, graph-based outputs could serve as a robust bridge between raw events and modern ML architectures.

If you want to preserve sparsity, you need tokens that mean something. Individual ON/OFF events are too atomic to be useful tokens, so a practical middle ground is a two‑stage model: a lightweight, streaming “tokenizer” that clusters local spatiotemporal activity into short‑lived micro‑features, followed by a stateful temporal model that reasons over those features. The tokenizer can be as simple as centroiding event bursts in a small spatial neighborhood with a short time constant, or as involved as a dynamic graph builder that fuses polarity, age, and motion cues. Either way, the goal is to transform a flood of spikes into a bounded, variable‑rate set of tokens with stable meaning. Next let's explore the type of models that work well with event camera data.

## Machine Learning Models
At their core, event cameras are change detectors, which means that we need memory in our machine learning models to remember where things were before they stopped moving. 
We can bake memory into the model architecture by using recurrence or attention. For example, [Recurrent Vision Transformers](https://openaccess.thecvf.com/content/CVPR2023/html/Gehrig_Recurrent_Vision_Transformers_for_Object_Detection_With_Event_Cameras_CVPR_2023_paper.html) and their variants maintain internal state across time and can handle temporally sparse inputs more naturally. These methods preserve temporal continuity, but there’s a catch: most of these methods still rely on dense, voxelized inputs. Even with more efficient [state-space models](https://openaccess.thecvf.com/content/CVPR2024/html/Zubic_State_Space_Models_for_Event_Cameras_CVPR_2024_paper.html) replacing LSTMs and BPTT (Backpropagation Through Time), we’re still processing a lot of zeros. Training is faster, but inference is still bottlenecked by inefficient representations.

Nowadays larger AI models are being pruned, distilled, and quantised to provide efficient edge models that can generalise well. Even TinyML models are [students](https://www.nature.com/articles/s41598-025-94205-9.pdf) of a larger model. We have to say goodbye to the idea of training tiny models from scratch for commercial event camera applications, because they won't perform well enough in the real world. 

Spiking neural networks (SNNs) are sometimes touted as a natural fit for event data. But in their traditional form, with binary activations and reset mechanisms, leaky integrate-and-fire (LIF) neurons are handcrafted biological abstractions. If we learned anything from machine learning, it's that handcrafted designs are inherently flawed. And neurons are an incredibly complex thing to model, as efforts such as [CZI’s Virtual Cells](https://chanzuckerberg.com/science/technology/virtual-cells/) and [DeepMind’s cell simulations](https://analyticsindiamag.com/ai-features/inside-google-deepminds-bold-vision-for-virtual-cell/) show. So let's not get hung up on the artificial neuron model itself, and instead use what works well, because the field is moving incredibly fast. 

I’m very optimistic about state space models (SSMs) for event vision. Instead of baking memory into heavy recurrence or dense attention, an SSM treats the scene’s latent dynamics as a continuous-time system and then discretizes only for inference. This means a single trained model can adapt to many operating modes: you can run it at different inference rates or even update state event-by-event with variable time steps—without retraining—simply by changing the integration step. That flexibility is a good match for sensors whose activity is unpredictable.

## Processors
Meyer et al. implemented a S4D SSM on Intel’s Loihi 2, constraining the state space to be diagonal so that each neuron evolves independently.
They mapped these one-dimensional state updates directly to Loihi’s programmable neurons and carefully placed layers to reduce inter-core communication, which resulted in much lower latency and energy use than a Jetson GPU in true online processing. 
I think it’s a compelling demonstration that SSMs can be run efficiently on stateful AI accelerator hardware and I'm curious what else is coming out of that. 

Some people argue that because event cameras output extremely sparse data, we can save energy by skipping zeros in the input or in intermediate activations. But I don't buy that argument because while the input might be much sparser than an RGB frame, the bulk of the computation actually happens in intermediate layers and works with higher level representations, which are hopefully similar for both RGB and event inputs. That means that in AI accelerators we can't exploit spatial event camera sparsity, and inference cost between RGB and event frames are essentially the same. Of course we might get different input frame rates / temporal sparsity, but those can be exploited on GPUs as well. 

Keep in mind that on mixed-signal hardware, rules are different. There's a breadth of new materials being explored, memristors and spintronics. The basic rule for analog is: if you need to convert from analog to digital too often, for error correction or because you're storing states or other intermediate values, your efficiency gains go out of the window. [Mythic AI](https://mythic.ai/) had to painfully learn that and [almost tanked](https://www.reddit.com/r/technology/comments/yvjgwu/analog_ai_chip_startup_mythic_runs_out_of_money/), and also [Rain AI](https://rain.ai/) pivoted from its original analog hardware and faces [an uncertain future](https://startupwired.com/2025/05/16/rain-ai-the-rise-and-fall-of-a-chipmaking-challenger/). The brain uses a mixture of analog (graded potentials, dendritic integration) and digital (spikes) signals and we can replicate this principle in silicon. But since the circuitry is the memory at the same time, it needs an incredible amount of space, and is organised in 3d. That's really costly to do in silicon, and the major challenge is getting the heat out, which is much easier in 2d. 

I think that the asynchronous compute principle is key for event cameras, but we need to realise that naïve asynchrony is not constructive. Think about a roundabout, and how it manages the flow of traffic without any traffic lights. When the traffic volume is low, every car is more or less in constant motion, and latency to cross the roundabout is minimal. As the volume of traffic grows, a roundabout becomes inefficient, because the movement of any car depends on the decisions of cars nearby. For high traffic flow, it becomes more efficient to use traffic lights to `batch process` the traffic for multiple lanes at once, which achieves the highest throughput of cars.
The same principle applies for events. When you have few pixels activated, you achieve the lowest latency when you process them as they come in, as in a roundabout. But as the amount of events / s gets larger, for example because you're moving the camera on a car or a drone, you need to get out the traffic lights, and start and stop larger batches of events. Ideally the size of the batch depends on the event rate. 

For more info about neuromorphic chips, I refer you to [Open Neuromorphic's Hardware Guide](https://open-neuromorphic.org/neuromorphic-computing/hardware/).

## Conclusion
Here are my main points:
* Event cameras won’t go mainstream until they move away from binary events and to richer output formats, whether from the sensor directly or an attached preprocessor.
* Event cameras follow the trajectory of other sensors that were developed and improved within the context of defence applications. 
* We need an efficient representation that is compatible with modern ML architectures. It might well be event frames in the end. 
* Keep it practical. Biologically-inspired approaches should not distract from deployment-grade ML solutions.

The recipe that scales is: build a token stream that carries meaning, train it with cross‑modal supervision and self‑supervision that reflects real sensor noise, keep a compact scene memory that is cheap to update, and make computation conditional on activity rather than on a fixed clock.

Deep learning started out with 32 bit floating point, dense representations, and neuromorphic started out on the other end of the spectrum at binary, extremely sparse representations. They are converging, with neuromorphic realising that binary events are expensive to transmit, and deep learning embracing 4 bit activations and 2:4 sparsity.

Interesting research directions for event cameras today are about dynamic graph representations for efficient tokenization, state space models for efficient inference, lossy compression for smaller file sizes. To unlock the full potential of event cameras, we need to solve the representation problem to make it compatible with modern deep learning hardware and software, while preserving the extreme sparsity of the data. 
Also we shouldn’t be too focused on biologically-inspired processing if we want this thing to scale anytime soon. I think that either the sensors must evolve to emit richer, token-friendly outputs, or they must be paired with dedicated pre-processors that produce high-level, potentially graph-based abstractions. Once that happens, event cameras become easy enough to work with to reach the mainstream. 

Ultimately, the application dictates the design. Gesture recognition does not need microsecond temporal resolution. Eye tracking doesn't need HD spatial resolution. And sometimes a motion sensor that will wake a standard camera will be the easiest solution. 

