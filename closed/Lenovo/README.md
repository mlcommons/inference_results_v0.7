# Lenovo Submission for MLPerf Inference v0.7 based on NVIDIA-Optimized Implementations

This is a repository of Lenovo submision of NVIDIA-optimized implementations for [MLPerf Inference Benchmark v0.7](https://www.mlperf.org/inference-overview/). For addtional details refer to NVIDIA's submission.


## Lenovo Submissions

This repository has the following submissions:

| Benchmark     | Datacenter Submissions                                        | Edge Submissions (Multistream may be optional)                                   |
|---------------|---------------------------------------------------------------|----------------------------------------------------------------------------------|
| 3D-UNET       | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline         | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, SingleStream              |
| BERT          | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, Server | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream                        |
| DLRM          | Accuracy: 99% and 99.9% of FP32<br>Scenarios: Offline, Server | None                                                                             |
| RNN-T         | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream                        |
| ResNet50      | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream, MultiStream           |
| SSD-MobileNet | None                                                          | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream, MultiStream           |
| SSD-ResNet34  | Accuracy: 99% of FP32<br>Scenarios: Offline, Server           | Accuracy: 99% of FP32<br>Scenarios: Offline, SingleStream, MultiStream           |

Benchmarks are stored in the [code/](code) directory.
Every benchmark contains a `README.md` detailing instructions on how to set it up.


## Lenovo Submission Systems

The systems that Lenovo submitted on are:

 - Datacenter systems
   - A100-PCIex4 (ThinkSystem SR670)
 - Edge systems
   - T4x1 (ThinkSystem SE350)

## Quick General Instructions

### Prerequisites

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- T4 or A100 NVIDIA GPUs
- NVIDIA Driver Version 450.xx or greater
- Ubuntu 18.04, other operating systems have not been tested.


### Before you run commands


- Run `export MLPERF_SCRATCH_PATH=<path/to/scratch/space>` 

- Run `make prebuild`.

This launches the Docker container (tag `mlperf-inference:<USERNAME>-latest`) with all the necessary packages installed. 
 The source codes in the repository are located at `/work` inside the docker image.

### Download and Preprocess Datasets and Download Models


- `ResNet50`: download [ImageNet 2012 Validation set](http://www.image-net.org/challenges/LSVRC/2012/) and unzip to `$MLPERF_SCRATCH_PATH/data/imagenet/`.
- `DLRM`: download [Criteo Terabyte dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) and unzip to `$MLPERF_SCRATCH_PATH/data/criteo/`.
- `3D-Unet`: download [BraTS 2019 Training set](https://www.med.upenn.edu/cbica/brats2019/registration.html) and unzip to `$MLPERF_SCRATCH_PATH/data/BraTS/MICCAI_BraTS_2019_Data_Training`.

Quick commands:

```
$ make download_model # Downloads models and saves to $MLPERF_SCRATCH_PATH/models
$ make download_data # Downloads datasets and saves to $MLPERF_SCRATCH_PATH/data
$ make preprocess_data # Preprocess data and saves to $MLPERF_SCRATCH_PATH/preprocessed_data
```

Notes:

- By default, the `make download_model`/`make download_data`/`make preprocess_data` commands run for all the benchmarks.
Add `BENCHMARKS=resnet50`, for example, to specify a benchmark.

### Running the repository


```
$ make build
```

```
$ make generate_engines RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> --config_ver=default,high_accuracy [OTHER FLAGS]"

```


#### Run harness on engines to get results

```
$ make run_harness RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> --config_ver=default,high_accuracy --test_mode=<AccuracyOnly/PerformanceOnly> [OTHER FLAGS]"
```



In the above, if `RUN_ARGS` is not specified, all supported benchmark-scenario pair will ecexute.
See [command_flags.md](command_flags.md) for more information on arguments.
The optimized engine files are saved to `/work/build/engines`.

Performance results will be printed to `stdout`, other logging to `stderr`.
LoadGen logs go to `/work/build/logs`.


### Run Calibration

See NVIDIA's calibration.md for an explanation on how calibration is used for this submission.

### Update Results

Refer to Nvidia's submission_guide.md on info how to populate  `results/` with the logs in `build/logs`.
Run the following command to update the LoadGen logs in:


### Run Compliance Tests and Update Compliance Logs

Refer to NVIDIA's submission_guide.md.

## Instructions for Auditors

Please refer to the `README.md` in each benchmark directory for auditing instructions.

## Other documentations:

- Refer to Nvidia submission documentation for full description of the benchmarks and results for additional systems.
