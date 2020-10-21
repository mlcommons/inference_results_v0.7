# MLPerf Inference v0.7 - OpenVINO via Collective Knowledge

We describe how to set up and run Intel's OpenVINO implementation (distributed
to partners in binary form) via automated, customizable and reproducible
[Collective Knowledge](http://cknowledge.org) (CK) workflows.

Contact info@dividiti.com if you have any questions.

# Table of Contents

1. [Setting up](#setting_up)
    1. [Basic setup](#setting_up_basic)
    1. [Image Classification](#setting_up_image_classification)
    1. [Object Detection](#setting_up_object_detection)
    1. [System-Under-Test](#setting_up_sut)
1. [Further info](#further_info)
    1. [Running experiments](#running)
    1. [Tuning performance](#tuning)

<a name="setting_up"></a>
# Setting up

<a name="setting_up_basic"></a>
## Basic setup

- Extract Intel's binary drop to e.g. your home directory (`$HOME`),
and point to it via an environment variable:

```bash
$ export CK_OPENVINO_DROP=$HOME/mlperf_ext_ov_cpp_v0.7-master
$ chmod u+x $CK_OPENVINO_DROP/Release/ov_mlperf
```

- Set up system-level dependencies via `sudo` (or as superuser):

```bash
$ sudo apt upgrade -y
$ sudo apt install -y gcc g++ python3 python3-pip python3-dev make cmake git wget zip libz-dev vim numactl
$ sudo apt clean
```

- Add `$HOME/.local/bin` to `PATH`:

```bash
$ echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
$ export PATH=$HOME/.local/bin:$PATH
```

- Install CK:

```bash
$ export CK_PYTHON=/usr/bin/python3
$ $CK_PYTHON -m pip install --ignore-installed pip setuptools --user
$ $CK_PYTHON -m pip install ck --user && ck version
```

- If you have access to dividiti's `ck-openvino-private` repository, pull it
  and dependent CK repositories (including `ck-openvino`, `ck-mlperf`,
`ck-env`) as follows:

```bash
$ ck pull repo --url=git@github.com:dividiti/ck-openvino-private
```

- Otherwise, pull the public `ck-openvino` repository and install the
  `ck-openvino-private` repository from an archived snapshot as follows:

```bash
$ ck pull repo:ck-openvino
$ ck add repo --zip=~/Downloads/ck-openvino-private.20200915.zip
```

- Detect Python 3:

```bash
$ ck detect soft:compiler.python --full_path=`which $CK_PYTHON` --quiet
```

- Use generic Linux settings with dummy frequency setting scripts:

```bash
$ ck detect platform.os --platform_init_uoa=generic-linux-dummy
```

<a name="setting_up_image_classification"></a>
## Image Classification

### Register the ImageNet 2012 validation dataset with CK

Unfortunately, the ImageNet validation dataset [can no longer](https://github.com/mlperf/inference_policies/issues/125) be automatically downloaded.
If you place a copy of this dataset e.g. under `/datasets/dataset-imagenet-ilsvrc2012-val/`, you can register it with CK as follows:

```bash
$ ck detect soft:dataset.imagenet.val \
--full_path=/datasets/dataset-imagenet-ilsvrc2012-val/ILSVRC2012_val_00000001.JPEG
```
The OpenVINO program also expects to find a copy of the labels in the same directory (otherwise segfaults):

```bash
$ ck install package --tags=dataset,imagenet,aux
$ cp `ck locate env --tags=dataset,imagenet,aux`/val.txt \
     /datasets/dataset-imagenet-ilsvrc2012-val/val_map.txt
```

### Register the ResNet50 OpenVINO model binary with CK

```bash
$ echo 0.7 | ck detect soft:model.openvino \
--full_path=$CK_OPENVINO_DROP/Models/resnet50/resnet50_int8.xml \
--extra_tags=image-classification,resnet,resnet50,quantized,int8,side.224 \
--ienv.ML_MODEL_MODEL_NAME=resnet50
```

### Run an accuracy test on 500 images (do not save results)

```bash
$ ck run cmdgen:benchmark.openvino-loadgen --verbose --no_record \
--sut=dellemc_r640xd6248r --model=resnet50 --target_qps=2500 \
--scenario=offline --mode=accuracy --dataset_size=500 \
--nvirtcpus=`grep -c processor /proc/cpuinfo` \
--nthreads={{{nvirtcpus}}} --nireq={{{nvirtcpus}}} --nstreams={{{nvirtcpus}}}
...
accuracy=75.400%, good=377, total=500
================================================================================
```

<a name="setting_up_object_detection"></a>
## Object Detection

### Install the COCO 2017 validation dataset

```bash
$ ck install package --tags=dataset,object-detection,coco.2017,val,original
```

### Register the SSD-ResNet34 OpenVINO model binary with CK

```bash
$ echo 0.7 | ck detect soft:model.openvino \
--full_path=$CK_OPENVINO_DROP/Models/ssd-resnet34/ssd-resnet34_int8.xml \
--extra_tags=object-detection,ssd-resnet,ssd-resnet34,quantized,int8,side.1200 \
--ienv.ML_MODEL_MODEL_NAME=ssd-resnet34 --ienv.ML_MODEL_USE_INV_MAP=1
```

### Run an accuracy test on 50 images (do not save results)

```bash
$ ck run cmdgen:benchmark.openvino-loadgen --verbose --no_record \
--sut=dellemc_r640xd6248r --model=ssd-resnet34 --target_qps=50 \
--scenario=offline --mode=accuracy --dataset_size=50 \
--nvirtcpus=`grep -c processor /proc/cpuinfo` \
--nthreads={{{nvirtcpus}}} --nireq={{{nvirtcpus}}} --nstreams={{{nvirtcpus}}}
...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.261
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.283
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.156
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.399
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.251
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.362
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.384
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.206
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.493
mAP=26.114%
================================================================================
```

<a name="setting_up_sut"></a>
## System-Under-Test

To fully benefit from the [automated submission preparation system](#submitting), you should prepare an instance of
[`module:sut`](https://github.com/dividiti/ck-mlperf/tree/master/module/sut) describing your System-Under-Test (SUT).
It is practically the same as `<system_desc_id>.json`, described in [Submission Rules 5.7](https://github.com/mlperf/policies/blob/master/submission_rules.adoc#57-system_desc_idjson-metadata).

So far, we have provided two examples:

- [`ck-mlperf:sut:dellemc_r740xd6248`](https://github.com/dividiti/ck-mlperf/blob/master/sut/dellemc_r740xd6248) (copied from [Dell's v0.5 submission](https://github.com/mlperf/inference_results_v0.5/blob/master/closed/DellEMC/systems/DELLEMC_R740xd6248_openvino-linux.json));
- [`ck-openvino-private:sut:dellemc_r640xd6248r`](https://github.com/dividiti/ck-openvino-private/tree/master/sut/dellemc_r640xd6248r).

Please verify that these files conform to the above Submission Rules.

To create a new SUT instance, you can copy one of the above entries **using CK** e.g.:

```bash
$ ck cp ck-openvino-private:sut:dellemc_r640xd6248r ck-openvino-private:sut:dellemc_r740xd8280m
```

and edit `.cm/meta.json` in the copy:

```bash
$ cd `ck find ck-openvino-private:sut:dellemc_r740xd8280m`
$ vim .cm/meta.json
...
```

The contents does not have to be 100% correct when running experiments, but will have to be when [preparing a submision package](#submitting).


<a name="further_info"></a>
# Further information

Please refer to the following READMEs for further information.
Contact info@dividiti.com if you have any questions.

<a name="running"></a>
## Running experiments

If you already know the optimal parameter values for your System-Under-Test
(SUT), you can re-run your experiments using CK, and then proceed to generating
a compliant submission package. Otherwise, you can use CK to [tune](#tune) them first.

### ResNet50

- (Closed) Offline: [`~/CK/ck-openvino-private/program/openvino-loadgen-v0.7-drop/README.resnet50-offline.md`](https://github.com/dividiti/ck-openvino-private/blob/master/program/openvino-loadgen-v0.7-drop/README.resnet50-offline.md)
- (Closed) Server: [`~/CK/ck-openvino-private/program/openvino-loadgen-v0.7-drop/README.resnet50-server.md`](https://github.com/dividiti/ck-openvino-private/blob/master/program/openvino-loadgen-v0.7-drop/README.resnet50-server.md)
- (Open) SingleStream: [`~/CK/ck-openvino-private/program/openvino-loadgen-v0.7-drop/README.resnet50-singlestream.md`](https://github.com/dividiti/ck-openvino-private/blob/master/program/openvino-loadgen-v0.7-drop/README.resnet50-singlestream.md)

### SSD-ResNet34

- (Closed) Offline: [`~/CK/ck-openvino-private/program/openvino-loadgen-v0.7-drop/README.ssd-resnet34-offline.md`](https://github.com/dividiti/ck-openvino-private/blob/master/program/openvino-loadgen-v0.7-drop/README.ssd-resnet34-offline.md)
- (Closed) Server: [`~/CK/ck-openvino-private/program/openvino-loadgen-v0.7-drop/README.ssd-resnet34-server.md`](https://github.com/dividiti/ck-openvino-private/blob/master/program/openvino-loadgen-v0.7-drop/README.ssd-resnet34-server.md)
- (Open) SingleStream: [`~/CK/ck-openvino-private/program/openvino-loadgen-v0.7-drop/README.ssd-resnet34-singlestream.md`](https://github.com/dividiti/ck-openvino-private/blob/master/program/openvino-loadgen-v0.7-drop/README.ssd-resnet34-singlestream.md)

<a name="tuning"></a>
## Tuning the performance

- [`~/CK/ck-openvino-private/program/openvino-loadgen-v0.7-drop/README.tuning.md`](https://github.com/dividiti/ck-openvino-private/blob/master/program/openvino-loadgen-v0.7-drop/README.tuning.md)
