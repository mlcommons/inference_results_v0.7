# Dependencies
## LoadGen
```shell script
git clone https://github.com/mlperf/inference.git mlperf_inference
cd mlperf_inference
git checkout r0.7
mkdir -p loadgen/build && cd loadgen/build
cmake ..
make
cp libmlperf_loadgen.a ..
```

## clipp - command line interfaces for modern C++
```bash
git clone https://github.com/muellan/clipp.git
cd clipp/
mkdir build
cd build/
cmake -DCMAKE_INSTALL_PREFIX=/opt/clipp ..
sudo make install
```

## spdlog - logging library
```
git clone https://github.com/gabime/spdlog.git
cd spdlog && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/spdlog .. && sudo make all install
```

# Networks
## ResNet50
### Compile model
```shell script
mkdir ~/models
cd quantize
bash compile.sh
cp resnet50.tpu ~/models
```

### Build Test System
```
mkdir -p build && cd build
cmake -DLOADGEN_PATH=~/mlperf_inference/loadgen -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 -DCMAKE_BUILD_TYPE=Release ..
make && sudo cp iva_resnet50 /usr/local/bin
```

### Single Stream
#### Performance
```
time iva_resnet50 SingleStream performance -d ~/datasets/imagenet -c ~/mlperf_inference/mlperf.conf -u ../conf/user.conf -v ~/datasets/imagenet/val_map.txt -m ~/models/resnet50.tpu
```
#### Accuracy
```
time iva_resnet50 SingleStream accuracy -d ~/datasets/imagenet -c ~/mlperf_inference/mlperf.conf -u ../conf/user.conf -v ~/datasets/imagenet/val_map.txt -m ~/models/resnet50.tpu
```
check accuracy
```
python ~/mlperf_inference/vision/classification_and_detection/tools/accuracy-imagenet.py  --imagenet-val-file ~/datasets/imagenet/val_map.txt --mlperf-accuracy-file mlperf_log_accuracy.json --dtype int32
```
#### Audit
```shell script
mkdir /tmp/compliance && cd /tmp/compliance
mkdir -p TEST01 TEST04-A  TEST04-B  TEST05

# Run TEST01 compliance check
cp ~/mlperf_inference/compliance/nvidia/TEST01/resnet50/audit.config TEST01
cd TEST01
iva_resnet50 SingleStream performance --dataset ~/datasets/imagenet -c ~/iva_mlperf/mlperf.conf -u ~/iva_mlperf/user.conf -v /home/m.moroz/datasets/imagenet/val_map.txt -m ~/models/resnet50.tpu
python ~/mlperf_inference/compliance/nvidia/TEST01/run_verification.py  --results_dir /tmp/mlperf-submission/closed/IVATech/results/iva-fpga-1/resnet/SingleStream --compliance_dir /tmp/compliance/TEST01 --output_dir /tmp/mlperf-submission/closed/IVATech/compliance/iva-fpga-1/resnet/SingleStream
cd -

for t in TEST04-A  TEST04-B  TEST05; do 
  cp ~/mlperf_inference/compliance/nvidia/$t/audit.config $t;
  cd $t;
  iva_resnet50 SingleStream performance --dataset ~/datasets/imagenet -c ~/iva_mlperf/mlperf.conf -u ~/iva_mlperf/user.conf -v /home/m.moroz/datasets/imagenet/val_map.txt -m ~/models/resnet50.tpu
  cd -
done
python ~/mlperf_inference/compliance/nvidia/TEST04-A/run_verification.py --test4A_dir TEST04-A --test4B_dir TEST04-B --output_dir /tmp/mlperf-submission/closed/IVATech/compliance/iva-fpga-1/resnet/SingleStream
python ~/mlperf_inference/compliance/nvidia/TEST05/run_verification.py  --results_dir /tmp/mlperf-submission/closed/IVATech/results/iva-fpga-1/resnet/SingleStream --compliance_dir /tmp/compliance/TEST05 --output_dir /tmp/mlperf-submission/closed/IVATech/compliance/iva-fpga-1/resnet/SingleStream
```
#### Truncate logs
```shell script
python ~/mlperf_inference/tools/submission/truncate_accuracy_log.py --input /tmp/mlperf-submission --output /tmp/mlperf-submission-truncated --submitter IVATech
```

#### Submission checker
```shell script
python ~/mlperf_inference/tools/submission/submission-checker.py --input /tmp/mlperf-submission-truncated --version v0.7 --submitter IVATech
```