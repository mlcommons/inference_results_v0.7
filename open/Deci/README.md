# Deci's Open Submission
Deci’s mission is to bridge the gap between ML model research/development and production solutions.
Our proprietary Automated Neural Architecture Construction (AutoNAC) technology automatically redesigns any deep neural network to be optimized for high throughput/low latency, in a hardware and data-aware fashion, while preserving the model’s original accuracy. Deci’s AutoNAC enables blazing-fast deep learning inference, on any hardware, unlocking new use cases at the edge and maximizing resource utilization on the cloud.

**An *AutoNAC-optimized Resnet-50* achieves up to 12x boost on-top of Intel’s OpenVino compilation!**

AutoNAC provides an extra 3.2x boost in comparison with an 8-bit quantized, Intel OpenVino Resnet50.

This super fast network can process ~160 images per second on a single core Intel's CPU!

We applied AutoNAC, on a laptop as well as Google Cloud Platform (GCP), to accelerate Resnet-50 with respect to three Intel CPU devices while preserving ImageNet top-1 accuracy:
- MacBook-pro 2019 - 1.4GHz quad-core Intel i5 CPU
- Low-cost CPU - 1 Intel Cascade Lake cores - GCP - n2-highcpu-2
- Standard CPU - 8 Intel Cascade Lake cores - GCP - n2-highcpu-16

Using AutoNAC Deci can accelerate any model for any hardware and data.\
Visit our website at [https://deci.ai](https://deci.ai) to learn more or to book a demo.

And, of course, stay tuned for our v0.8 submission...

Executive Summary:
|         Hardware           | Latency (ms) | Throughput (imgs/sec) | Boost compared to Resnet50 |
|:--------------------------:|:------------:|:---------------------:|:--------------------------:|
|     MacBook pro 2019       |      7.0     |         204           | 6.8-12x                    |
|  1 Intel CascadeLake cores |      6.4     |         148           |   10.6x                    |
| 8 Intel CascadeLake cores  |      2.1     |         1092          | 5-9.5x                     | 


*To reproduce, see instructions in the `open/deci/code/reproduce_deci_instructions.md` or contact [sefi@deci.ai](sefi@deci.ai)*
