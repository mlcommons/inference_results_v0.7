#!/bin/bash

export RTE_ACQUIRE_DEVICE_UNMANAGED=1
export XILINX_XRT=/opt/xilinx/xrt 

usage() {
  echo -e ""
  echo "Usage:"
  echo "------------------------------------------------"
  echo -e " ./run.sh " 
  echo -e "          --mode <mode-of-exe>" 
  echo -e "          --scenario <mlperf-screnario>"
  echo -e "          --dir <image-dir>"
  echo -e "          --nsamples <number-of-samples>"
  echo -e ""
  echo "With Runner interface:"
  echo -e " ./run.sh --mode AccuracyOnly --scenario SingleStream"
  echo -e ""
}
# Default
MODE=PerformanceOnly
DIRECTORY=${HOME}/CK-TOOLS/dataset-imagenet-ilsvrc2012-val
SCENARIO=Server
SAMPLES=1024
TARGET_QPS=4000
MAX_ASYNC_QUERIES=200

RT_ENGINE=${RT_ENGINE:=/home/demo/aaronn/rt-engine}
MLP_INFER_ROOT=${MLP_INFER_ROOT:=/home/demo/aaronn/mlperf-inference}
DPU_DIR=${DPU_DIR:=model.dpuv4e/meta.json}

# Parse Options
while true
do
  if [[ -z "$1" ]]; then break; fi
  case "$1" in
    -m  |--mode               ) MODE="$2"              ; shift 2 ;;
    -d  |--dir                ) DIRECTORY="$2"         ; shift 2 ;;
    -s  |--scenario           ) SCENARIO="$2"          ; shift 2 ;;
    -n  |--nsamples           ) SAMPLES="$2"           ; shift 2 ;;
    -r  |--qps                ) TARGET_QPS="$2"        ; shift 2 ;;
    -a  |--max_async_queries  ) MAX_ASYNC_QUERIES="$2" ; shift 2 ;;
    -h  |--help               ) usage                  ; exit  1 ;;
     *) echo "Unknown argument : $1";
        exit 1 ;;
  esac
done

OPTIONS="--mode ${MODE} --scenario ${SCENARIO} --num_samples ${SAMPLES} --max_async_queries ${MAX_ASYNC_QUERIES} --qps ${TARGET_QPS}"

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib:${RT_ENGINE}/build:/opt/xilinx/xrt/lib:${MLP_INFER_ROOT}/loadgen/build:/usr/local/lib64/:/usr/local/lib
./app.exe --dpudir ${DPU_DIR} --imgdir ${DIRECTORY} ${OPTIONS} 

if [ "${MODE}" == "AccuracyOnly" ]
then
python3 ${MLP_INFER_ROOT}/vision/classification_and_detection/tools/accuracy-imagenet.py --imagenet-val-file=${DIRECTORY}/val_map.txt --mlperf-accuracy-file=mlperf_log_accuracy.json |& tee accuracy.txt
fi
