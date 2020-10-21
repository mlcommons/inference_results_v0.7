# MXNet ResNet50-v1.5 for MLPerf v0.7 inference

## 1. Description

This folder contains a model converter from ONNX to MXNet, a quantization script to quantize FP32 model to INT8 model, as well as the implementations to support running Offline and Server scenarios which compliance to MLPerf requirements.
The calibration dataset was based on mlperf provided list in [here](https://github.com/mlperf/inference/blob/master/calibration/ImageNet/cal_image_list_option_1.txt)

## 2. Python Environment and LoadGen Setup
### 2.1 Python Environment Setup
```bash
sudo apt install g++
sudo apt install libopencv-dev python3-opencv
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
chmod +x Anaconda3-2020.02-Linux-x86_64.sh
bash ./Anaconda3-2020.02-Linux-x86_64.sh
conda create -n resnet50 python=3.7.7
source activate resnet50
pip install cmake
pip install absl-py
```

### 2.2 LoadGen Installation
Require g++ version >= 5.0 to support c++14, refer [here](https://github.com/mlperf/inference/blob/master/loadgen/README_BUILD.md) for more detail guide.
```bash
git clone --recurse-submodules https://github.com/mlperf/inference.git mlperf_inference
cd mlperf_inference/loadgen
CFLAGS="-std=c++14" python setup.py develop
cd ../..
```

## 3. Convert Model From ONNX to MXNet
### 3.1 Install MKL, MXNet, ONNX
```bash
# Install MKL
sudo bash
# <type your user password when prompted.  this will put you in a root shell>
# cd to /tmp where this shell has write permission
cd /tmp
# now get the key:
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# now install that key
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# now remove the public key file exit the root shell
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
exit
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt update
sudo apt install intel-mkl-2019.5-075

# Add the path for libiomp5.so to make sure this lib is able to be found.
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:$LD_LIBRARY_PATH
```

```bash
# Install MXNet
git clone https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
git checkout 6ae469a17ebe517325cdf6acdf0e2a8b4d464734
git submodule update --init
make -j USE_OPENCV=0 USE_MKLDNN=1 USE_BLAS=mkl USE_PROFILER=0 USE_LAPACK=0 USE_GPERFTOOLS=0 USE_INTEL_PATH=/opt/intel/
cd python && python setup.py install
cd ../..
```

```bash
# Install ONNX
conda install -c conda-forge protobuf=3.9 onnx
pip install opencv-python pycocotools onnxruntime
```

### 3.2 Convert Model
```bash
git clone https://gitlab.devtools.intel.com/mlperf/mlperf-inference-v0.7-intel-submission.git mlperf-inf-intel
cd mlperf-inf-intel/closed/Intel/code/resnet/resnet-mx

# Create model folder to store MXNet models
mkdir model

# Download ONNX model
wget -O ./model/resnet50-v1.5.onnx https://zenodo.org/record/2592612/files/resnet50_v1.onnx

# Convert to MXNet
python tools/onnx2mxnet.py
```
The converted FP32 model for MXNet is located at: `model/resnet50_v1b-symbol.json` and `model/resnet50_v1b-0000.params`.

## 4. Quantize FP32 model to INT8 model
To get INT8 model you would need to quantize the model using calibration dataset and quantization tool.

### 4.1 Prepare Calibration Dataset
The calibration dataset (image list) is from [mlperf](http://github.com/mlperf/inference/blob/master/calibration/ImageNet/cal_image_list_option_1.txt).

You can also find it in in the following path: `mlperf_inference/calibration/ImageNet/cal_image_list_option_1.txt`.

The preprocess to the dataset will be executed in the calibration stage, which will generate both FP32 and INT8 datatype numpy array in the folder of `preprocessed/imagenet/NCHW/`. Only the image in the calibration dataset list will be used when doing the calibration.

### 4.2 Quantization Tool Installation
Intel® Low Precision Optimization Tool is used to quantize the FP32 model, refer [here](https://github.com/intel/lp-opt-tool) for more detail information.

Follow the instructions to install Intel® Low Precision Optimization Tool:
```bash
git clone https://github.com/intel/lp-opt-tool
cp ilit_calib.patch lp-opt-tool/
cd lp-opt-tool && git checkout c468259 && git apply ilit_calib.patch
python setup.py install
cd ..
```

### 4.3 Quantize Model With Calibration Dataset
Quantize and calibrate the model by `calib.sh`.

```bash
# update the following path based on your env
export DATASET_PATH=/lustre/dataset/imagenet/img_raw/ILSVRC2012_img_val
export DATASET_LIST=./val_map.txt
export CALIBRATION_IMAGE_LIST=./cal_image_list_option_1.txt
./calib.sh
```
After quantization, the INT8 model is located at: `model/resnet50_v1b-quantized-symbol.json` and `model/resnet50_v1b-quantized-0000.params`.

## 5. Run Performance and Accuracy Test
### 5.1 Copy The File of mlperf.conf
```bash
cp mlperf_inference/mlperf.conf mlperf-inf-intel/closed/Intel/code/resnet/resnet-mx
```

### 5.2 Run Offline/Server Scenario
```bash
# update the following path based on your env
export DATASET_PATH=/lustre/dataset/imagenet/img_raw/ILSVRC2012_img_val
export DATASET_LIST=./val_map.txt

export MODEL_PREFIX=model/resnet50_v1b-quantized

# run offline scenario
export BATCH_SIZE=64
export CPUS_PER_INSTANCE=1
export NUM_INSTANCE=56
./run.sh offline

# run offline with accuracy scenario
export BATCH_SIZE=64
export CPUS_PER_INSTANCE=2
export NUM_INSTANCE=28
./run.sh offline accuracy

# run server scenario
export BATCH_SIZE=1
export CPUS_PER_INSTANCE=2
export NUM_INSTANCE=28
./run.sh server

# run server with accuracy scenario
export BATCH_SIZE=1
export CPUS_PER_INSTANCE=2
export NUM_INSTANCE=28
./run.sh server accuracy
```
