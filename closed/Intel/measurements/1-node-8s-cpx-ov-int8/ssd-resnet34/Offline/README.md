# Instructions for using OpenVino for MLPerf
## Requirements
+ OS: Ubuntu (Tested on 20.04 only).
+ GCC (Tested with 7.5)
+ cmake (Tested with 3.17.2)
+ Python (Tested with 3.6)
+ [OpenCV](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) (Tested with 4.3.0)
+  Build gflags

    ```git clone https://github.com/gflags/gflags.git```
    
    ```cd gflags```
    
    ```mkdir build && cd build```
    
    ```cmake ..```
    
    ```make ```
    
  + Build boost_filesystem.so 
    
    ```wget https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz

    tar -xzvf boost_1_72_0.tar.gz
    
    cd boost_1_72_0
    
    ./bootstrap --with-libraries=filesystem && ./b2 --with-filesystem
    
## Build steps

### Build loadgen library

Follow instructions from https://github.com/mlperf/inference/blob/master/loadgen/README_BUILD.md

  
## Download and build OpenVino

Clone and build OpenVino (https://github.com/openvinotoolkit/openvino/tree/pre.2021.1) with OpenMP: 

**NB**: In the cmake command sub ```/path/to/opencv/build``` with where you built OpenCV

Follow build steps in https://github.com/openvinotoolkit/openvino/tree/pre.2021.1/build-instruction.md#build-steps.

CMAKE Command to build with OMP:

    cmake -DENABLE_VPU=OFF -DENABLE_CLDNN=OFF -DENABLE_GNA=OFF -DENABLE_DLIA=OFF -DENABLE_TESTS=OFF -DENABLE_VALIDATION_SET=OFF -DTHREADING=OMP **-DNGRAPH_ONNX_IMPORT_ENABLE=OFF -DNGRAPH_DEPRECATED_ENABLE=FALSE** ../

### Build Intel OpenVINO mlperf code

Edit appropriate entries in build-ovmlperf.sh (Check library paths)
    
    ./build-ovmlerf.sh

###
## For Using Quantized models
Please refer to closed/Intel/calibration/OpenVINO in this submission repository.

## Steps to Run (except for 3D-UNET)

1. To download dataset and annotation files for **Resnet50** (ImageNet dataset) and **SSD-Mobilenet** (COCO dataset) follow instructions on https://github.com/mlperf/inference/tree/master/vision/classification_and_detection#datasets. 
To download dataset for **Bert** - Download Squad dataset from  Google Bert repo https://github.com/allenai/bi-att-flow/raw/master/squad/evaluate-v1.1.py, https://github.com/rajpurkar/SQuAD-explorer/blob/master/dataset/dev-v1.1.json. Path to this directory would be set as "--data_path" parameter in script.
Copy vocab.txt from this repo to Squadv1.1 dataset folder.
2. For Bert run, place dev_v1.1.json and vocab.txt in py-bindings\data before run.
3. Run scripts for Resnet50, SSD-Resnet34 and Bert from "scripts" directory. Please set appropriate (data, model, configs etc) paths in script. E.g. ```./scripts/bert-singlestream.sh``` which runs bert in singlestream mode. 
    
    BERT

        SingleStream: ```./scripts/bert-singlestream.sh```

        Offline: ```./scripts/bert-offline.sh```

        Server: ```./scripts/bert-server.sh```

    ResNet-50

        SingleStream: ```./scripts/resnet50-singlestream.sh```

        Offline: ```./scripts/resnet50-offline.sh```

        Server: ```./scripts/resnet50-server.sh``

    SSD-ResNet34

        SingleStream: ```./scripts/ssd-resnet34-singlestream.sh```

        Offline: ```./scripts/ssd-resnet34-offline.sh```

        Server: ```./scripts/ssd-resnet34-server.sh``
## Known issues
* Issue:
terminate called after throwing an instance of 'InferenceEngine::details::InferenceEngineException'
 what():  can't protect
Solution:
Patch with the following your current and any further submission machines as root:
 1. Add the following line to **/etc/sysctl.conf**: 
 vm.max_map_count=2097152 
  
 2. You may want to check that current value is
 too small with `cat /proc/sys/vm/max_map_count` 
  
 3. Reload the config as
 root: `sysctl -p` (this will print the updated value)
