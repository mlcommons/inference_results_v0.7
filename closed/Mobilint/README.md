# MLPerf Submission: Mobilint Accelerator
This document describes the details of Mobilint's AI Accelerator for MLPerf Inference v0.7 submission. The details consist of brief information of benchmark systems, system requirements, and lastly step-by-step instructions to run the benchmark.

## Overview
Benchmark for Mobilint Accelerator mainly consists of internal model compiler, preprocessor, and System Under Test. A user can compile the model using the compiler, upload the instructions into the accelerator, upload the preprocessed data, and finally get the result from the accelerator.

## Benchmark Summary
In this round v0.7, Mobilint submits the benchmark result as below:
| System | Division | Scenario | Model |
| ------ | ----- | ------ | ------ |
| Mobilint Edge | Closed | SingleStream, Offline | Resnet50-v1.5, SSD-MobileNet-v1 |
| Mobilint Edge-Five | Open | Offline | Resnet50-v1.5, SSD-MobileNet-v1 |

For the quality (Accuracy) of the SUT per models are:
| Model | Source model | Measured accuracy |
| ----- | ------ | ------ |
| SSD-MobileNet-v1 | ssd-mobilenet 300x300, ONNX   | 23.028% mAP |
| ResNet50-v1.5 | resnet-50-v1.5, ONNX  | 76.318% |

## System Description
### General System Configurations
* Ubuntu 18.04.3
* Python 3.6.9
### Binaries/Libraries Information
* Loadgen (tag v0.7.1; libmlperf_loadgen.a)
  * MD5 Checksum : f48bdb81f162ad030725c7e9c9fe794e  libmlperf_loadgen.a
* Xilinx DMA Device driver (xdma.ko)
  * MD5 Checksum : 677c32446ad39303d4e8e5d2c87fe238 xdma.ko
* Private acceleration library (libmaccel.so)
  * MD5 Checksum : f0c4ddcf3545d6140d07faf9f4f33f89  libmaccel.so

## Usage
### General Steps
General benchmark steps are following:
1. Compile the model using the internal compiler and generate `imem.bin`, `lmem.bin`, `dmem.bin`, `ddr.bin` per model.
2. Preprocess the dataset using `preprocess_resnet.py` or `preprocess_imagenet.py` appropriate for the dataset to be tested.
3. Extract only paths from `instances_val2017.json` or `val_map.txt` using `extract_dataset_list.py`.
4. Run the SUT (Scenario, Mode, Model, Dataset are the arguments)

```bash
# insmod xdma.ko
$ cp libmaccel.so /usr/lib
$ make
$ python extract_dataset_list.py --dataset-name="DATASET_NAME" --output="OUTPUT_FILENAME" --input="INPUT_FILENAME" --root-path="ROOT_PATH_TO PREPROCESSED_DATA"
$ ./benchmark --scenario="SCENARIO" --mode="MODE" --model="MODEL" --dataset="PATH_TO_VAL_MAP_OR_INSTANCES_VAL2017_PREPROCESSED"
```
### Dataset Path Extractor Arguments
|dataset-name|output|input|root-path|
|-----------|------|-----|------|
|COCO, ImageNet| Output filename | Input file path (`val_map.txt` for ImageNet, `instances_val2017.json` for COCO) | Root path to the preprocessed Image data |

### Benchmark SUT Arguments
|model|scenario|mode|dataset|
|----|-------|----|------|
|SSD-MobileNets-v1, Resnet50-v1.5 (But please use the code under the model)|SingleStream, MultiStream, Server, Offline|AccuracyOnly, PerformanceOnly|Path to preprocessed val_map.txt (ResNet50) or instances_val2017.json (SSD-MobileNet); the output of `Step 3`|

