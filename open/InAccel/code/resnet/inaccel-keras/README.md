# InAccel MLPerf Inference Benchmarks for Image Classification

This is the reference implementation for InAccel MLPerf Inference benchmarks.

## Supported Models

| model       | framework                                         | accuracy | dataset                 | model link                           | model source         | precision | notes                                                                                                         |
| ----------- | ------------------------------------------------- | -------- | ----------------------- | ------------------------------------ | -------------------- | --------- | ------------------------------------------------------------------------------------------------------------- |
| resnet50-v1 | [inaccel-keras](https://github.com/inaccel/keras) | 62.266%  | imagenet2012 validation | [from xilinx](http://bit.ly/2PsHFfE) | Xilinx Research Labs | int8      | NHWC. More information on quantized resnet50 v1 can be found [here](https://github.com/Xilinx/ResNet50-PYNQ). |

## Prerequisites and Installation

Setup InAccel in 5 minutes: https://docs.inaccel.com

Install the InAccel backend:

```sh
pip install inaccel-keras
```

Get the accelerators for Xilinx Alveo U250:

```sh
inaccel bitstream install https://store.inaccel.com/artifactory/bitstreams/xilinx/u250/xdma_201830.2/xilinx/com/researchlabs/1.1/1resnet50
```

Build and install the benchmark:

```sh
make all
```

## Running the benchmark

### One time setup

Download the model and dataset for the model you want to benchmark.

```sh
wget -P YourModelFileLocation https://inaccel-demo.s3.amazonaws.com/models/resnet50_weights.h5
```

Both local and docker environment need to set 2 environment variables:

```sh
export MODEL_DIR=YourModelFileLocation
export DATA_DIR=YourImageNetLocation
```

### Run local

```sh
cd inference/vision/classification_and_detection
./run_local.sh inaccel resnet50 fpga
```
