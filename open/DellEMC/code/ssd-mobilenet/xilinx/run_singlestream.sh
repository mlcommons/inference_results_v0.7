#!/bin/bash
#export MLPERF_INFERENCE=/proj/rdi/staff/gaoyue/gaoyue/mlperf_inference
#export MLPERF_INFERENCE=/scratch/mlperf/mlperf_inference
#export IMGDIR=/proj/rdi/staff/gaoyue/dataset-coco-2017-val
#export IMGDIR=/wrk/dcgmktg_bench_xhd/amardeep/datasets/COCO_Dataset


./app.exe --scenario SingleStream  --num_queries 1024 --num_samples 256  --min_time 60000 --dpudir ${1} --imgdir ${IMGDIR} --mode PerformanceOnly
#./app.exe --scenario SingleStream  --num_queries 8 --min_time 10000 --dpudir ${1} --imgdir ${IMGDIR} --mode PerformanceOnly

#python ${MLPERF_INFERENCE}/v0.5/classification_and_detection/tools/accuracy-imagenet.py --imagenet-val-file=${IMGDIR}/val_map.txt --mlperf-accuracy-file=mlperf_log_accuracy.json
