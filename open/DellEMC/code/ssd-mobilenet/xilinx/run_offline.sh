#!/bin/bash

#export MLPERF_INFERENCE=/proj/rdi/staff/gaoyue/gaoyue/mlperf_inference
#export IMGDIR=/proj/rdi/staff/gaoyue/dataset-coco-2017-val
./app.exe --scenario Offline --num_queries 1 --num_samples 256  --min_time 60000  --max_async_queries  1 --qps 4000  --batch 1 --thread_num 1  --dpudir ${1} --imgdir ${IMGDIR} --mode PerformanceOnly

#python ${MLPERF_INFERENCE}/v0.5/classification_and_detection/tools/accuracy-imagenet.py --imagenet-val-file=${IMGDIR}/val_map.txt --mlperf-accuracy-file=mlperf_log_accuracy.json
