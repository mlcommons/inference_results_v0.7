## Instructions for Using TensorFlow for MLPerf Inference
* Follow the instruction in closed/Intel/code/resnet-tf/README.md to build
TensorFlow and loadgen integration

* Run the following command for performance
```
cd loadrun
./run_loadrun.sh 28 offline 28 resnet50 56 PerformanceOnly
```
* Run the following command for accuracy
```
cd loadrun
./run_loadrun.sh 28 offline 28 resnet50 56 AccuracyOnly
```
