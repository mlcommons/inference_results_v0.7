# MLPerf Inference v0.7 NVIDIA-Optimized Implementations

This is a repository of NVIDIA-optimized implementations for [MLPerf Inference Benchmark v0.7](https://www.mlperf.org/inference-overview/).

## Benchmarks

The following *benchmarks* are part of our submission for MLPerf Inference v0.7:

 - **3D-Unet** (3d-unet)
 - **BERT** (bert)
 - **DLRM** (dlrm)
 - **RNN-T** (rnnt)
 - **ResNet50** (resnet50)
 - **SSD-MobileNet** (ssd-mobilenet)
 - **SSD-ResNet34** (ssd-resnet34)

## Scenarios

Each of the above benchmarks can run in one or more of the following four inference *scenarios*:

 - **SingleStream**
 - **MultiStream**
 - **Offline**
 - **Server**

Please refer to the [MLPerf Inference official page](https://www.mlperf.org/inference-overview/) for explanations about the scenarios.

## NVIDIA Submissions

Our MLPerf Inference v0.7 implementation has the following submissions:

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
Every benchmark contains a `README.md` detailing instructions on how to set up that benchmark, including:

 - Downloading the dataset and model
 - Running any necessary preprocessing
 - Details on the optimizations being performed

For details on how to run each benchmark, see below.

## NVIDIA Submission Systems

The systems that NVIDIA supports and has tested are:

 - Datacenter systems
   - A100-SXM4x8 (DGX-A100)
   - A100-PCIex2
   - T4x8
   - T4x20
 - Edge systems
   - A100-SXM4x1
   - A100-SXM4x1 - MIG-1x1g.5gb
   - A100-PCIex1
   - T4x1
   - AGX Xavier
   - Xavier NX

## General Instructions

:warning: **IMPORTANT**: Please use [closed/NVIDIA](closed/NVIDIA) (this directory) as the working directory when running any of the commands below. :warning:

**Note:** Inside the Docker container, [closed/NVIDIA](closed/NVIDIA) will be mounted at `/work`.

If you are working on the MLPerf Inference open submission, use [open/NVIDIA](open/NVIDIA) instead.

This section describes the steps needed to run harness with default configurations, weights, and validation datasets on NVIDIA submission systems to reproduce.
Please refer to later sections for instructions on auditing.

### Prerequisites

For x86_64 systems:

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- Turing or Ampere-based NVIDIA GPUs
- NVIDIA Driver Version 450.xx or greater

We recommend using Ubuntu 18.04.
Other operating systems have not been tested.

For Jetson Xavier:

- [20.09 Jetson CUDA-X AI Developer Preview](https://developer.nvidia.com/embedded/20.09-Jetson-CUDA-X-AI-Developer-Preview)
- Dependencies can be installed by running this script: [install_xavier_dependencies.sh](scripts/install_xavier_dependencies.sh). Note that this might take a while, on the order of several hours.

### Before you run commands

Before running any commands detailed below, such as downloading and preprocessing datasets, or running any benchmarks, you should
set up the environment by doing the following:

- Run `export MLPERF_SCRATCH_PATH=<path/to/scratch/space>` to set your scratch space path.
We recommend that the scratch space has at least **3TB**.
The scratch space will be used to store models, datasets, and preprocessed datasets.

- For x86_64 systems (not Xavier): Run `make prebuild`.
This launches the Docker container with all the necessary packages installed. 

 The docker image will have the tag `mlperf-inference:<USERNAME>-latest`.
 The source codes in the repository are located at `/work` inside the docker image.

### Download and Preprocess Datasets and Download Models

Each [benchmark](code) contains a `README.md` that explains how to download and set up the dataset and model for that benchmark.
The following commands allow you to download all datasets and models, and preprocesses them, except for downloading the datasets needed by `ResNet50`, `DLRM`, and `3D-Unet` since they don't have publicly available download links.

- For `ResNet50`, please first download the [ImageNet 2012 Validation set](http://www.image-net.org/challenges/LSVRC/2012/) and unzip the images to `$MLPERF_SCRATCH_PATH/data/imagenet/`.
- For `DLRM`, please first download the [Criteo Terabyte dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) and unzip the files to `$MLPERF_SCRATCH_PATH/data/criteo/`.
- For `3D-Unet`, please first download the [BraTS 2019 Training set](https://www.med.upenn.edu/cbica/brats2019/registration.html) and unzip the data to `$MLPERF_SCRATCH_PATH/data/BraTS/MICCAI_BraTS_2019_Data_Training`.

Quick commands:

```
$ make download_model # Downloads models and saves to $MLPERF_SCRATCH_PATH/models
$ make download_data # Downloads datasets and saves to $MLPERF_SCRATCH_PATH/data
$ make preprocess_data # Preprocess data and saves to $MLPERF_SCRATCH_PATH/preprocessed_data
```

Notes:

- The combined preprocessed data can be huge.
- Please reserve at least **3TB** of storage in `$MLPERF_SCRATCH_PATH` to ensure you can store everything.
- By default, the `make download_model`/`make download_data`/`make preprocess_data` commands run for all the benchmarks.
Add `BENCHMARKS=resnet50`, for example, to specify a benchmark.

### Running the repository

Running models is broken down into 3 steps:

#### Build

Builds the required libraries and TensorRT plugins:

```
$ make build
```

#### Generate TensorRT engines

```
$ make generate_engines RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> --config_ver=default,high_accuracy [OTHER FLAGS]"
```

If `RUN_ARGS` is not specified, all engines for each supported benchmark-scenario pair will be built.
See [command_flags.md](command_flags.md) for information on arguments that can be used with `RUN_ARGS`.
The optimized engine files are saved to `/work/build/engines`.

#### Run harness on engines

:warning: **IMPORTANT**: The DLRM harness requires around **40GB** of free CPU memory to load the dataset.
Otherwise, running the harness will crash with `std::bad_alloc`. :warning:

```
$ make run_harness RUN_ARGS="--benchmarks=<BENCHMARKS> --scenarios=<SCENARIOS> --config_ver=default,high_accuracy --test_mode=<AccuracyOnly/PerformanceOnly> [OTHER FLAGS]"
```

If `RUN_ARGS` is not specified, all harnesses for each supported benchmark-scenario pair will be run.
See [command_flags.md](command_flags.md) for `RUN_ARGS` options.
Note that if an engine has not already been built for a benchmark-scenario pair (in the earlier step), this will result in an error.

The performance results will be printed to `stdout`.
Other logging will be sent to `stderr`.
LoadGen logs can be found in `/work/build/logs`.

### Notes on runtime and performance

- MultiStream scenario takes a long time to run (4-5 hours) due to the minimum query count requirement. If you would like to run it for shorter runtime, please add `--min_query_count=1` to `RUN_ARGS`.
- To achieve maximum performance for Server scenario, please set Transparent Huge Pages (THP) to *always*.
- To achieve maximum performance for Server scenario on T4x8 and T4x20 systems, please lock the clock at the max frequency by `sudo nvidia-smi -lgc 1590,1590`.
- As a shortcut, doing `make run RUN_ARGS="..."` will run `generate_engines` and `run_harness` in succession. If multiple benchmark-scenario pairs are specified, engines will only run after all engines are successfully built.
- If you get INVALID results, or if the test takes a long time to run, or for more performance tuning guidance, or if you would like to run the benchmarks on an unsupported GPU, please refer to the [performance_tuning_guide.md](performance_tuning_guide.md).

### Run code on Multi Instance GPU (MIG) slice

The repository only supports running on a single 1x1g.5gb MIG slice. Any other partitioning will require adding a new submission MIG system with new parameters.

1. Enable MIG on the desired GPU with `sudo nvidia-smi -mig 1 -i $GPU`
2. Run scripts/mig_configure.sh to create MIG instances. This script creates seven 1g.5gb instances. For other profiles, create the instances with custom commands in this step.
3. Run `make prebuild`
4. Once inside the container run `source scripts/mig_set_device.sh` this selects one of the instances at random to run the workload.
5. Run `make run/generate_engines ..` as normal
6. Before exiting the machine run `scripts/mig_teardown.sh` if possible. This destroys the instances created in step 2.
7. Disable MIG with `sudo nvidia-smi -mig 0 -i $GPU`

### Run code in Headless mode

If you would like to run the repository without launching the interactive docker container, you can run `make build_docker` to build the container without launching an interactive shell.

To open a shell for an already built Docker container: `make attach_docker`.

To run commands inside an already built Docker container:

```
docker run -dt -e NVIDIA_VISIBLE_DEVICES=ALL -w /work \
    --security-opt apparmor=unconfined --security-opt seccomp=unconfined \
    -v $HOME:/mnt$HOME \
    -v $PWD:/work \
    -v $MLPERF_SCRATCH_PATH:$MLPERF_SCRATCH_PATH \
    -e MLPERF_SCRATCH_PATH=$MLPERF_SCRATCH_PATH \
    --name mlperf-inference-<USERNAME> mlperf-inference:<USERNAME>-latest
docker exec mlperf-inference-<USERNAME> bash -c '<COMMAND1>'
docker exec mlperf-inference-<USERNAME> bash -c '<COMMAND2>'
docker exec mlperf-inference-<USERNAME> bash -c '<COMMAND3>'
...
docker container stop mlperf-inference-<USERNAME>
docker container rm mlperf-inference-<USERNAME>
```

### Run Calibration

The calibration caches generated from default calibration set are already provided in each benchmark directory.

If you would like to re-generate the calibration cache for a specific benchmark, please run the following command:

```
$ make calibrate RUN_ARGS="--benchmarks=<BENCHMARK>"
```

See [calibration.md](calibration.md) for an explanation on how calibration is used for NVIDIA's submission.

### Update Results

Run the following command to update the LoadGen logs in `results/` with the logs in `build/logs`:

```
$ make update_results
```

Please refer to [submission_guide.md](submission_guide.md) for more detail about how to populate the logs requested by
MLPerf Inference rules under `results/`.

### Run Compliance Tests and Update Compliance Logs

Please refer to [submission_guide.md](submission_guide.md).

## Instructions for Auditors

Please refer to the `README.md` in each benchmark directory for auditing instructions.

## Other documentations:

- [FAQ.md](FAQ.md): Frequently asked questions.
- [performance_tuning_guide.md](performance_tuning_guide.md): Instructions about how to run the benchmarks on your systems using our code base, solve the INVALID result issues, and tune the parameters for better performance.
- [submission_guide.md](submission_guide.md): Instructions about the required steps for a valid MLPerf Inference submission with or without our code base.
- [calibration.md](calibration.md): Instructions about how we did post-training quantization on activations and weights.
- [command_flags.md](command_flags.md): List of some flags supported by `RUN_ARGS`.
- [Per-benchmark READMEs](code/README.md): Instructions about how to download and preprocess the models and the datasets for each benchmarks and lists of optimizations we did for each benchmark.
