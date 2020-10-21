## Instructions for Using TensorFlow for MLPerf Inference
* Follow the instruction in closed/Intel/code/resnet-tf/README.md to build
TensorFlow and loadgen integration

* Run the following command for performance
```
cd loadrun
NUM_INTRA_THREADS=7 ./run_loadrun.sh 28 offline 32 resnet50 224 PerformanceOnly
```
* Run the following command for accuracy
```
cd loadrun
NUM_INTRA_THREADS=7 ./run_loadrun.sh 28 offline 32 resnet50 224 AccuracyOnly
```
