#!/bin/bash

#export MLPERF_INFERENCE=/proj/rdi/staff/gaoyue/gaoyue/mlperf_inference
#export IMGDIR=/proj/rdi/staff/gaoyue/dataset-coco-2017-val

./app.exe --scenario Offline --num_samples 5000 --num_queries 1  --min_time 60000  --max_async_queries  1 --qps 1000  --batch 1 --thread_num 1  --dpudir ${1} --imgdir ${IMGDIR} --mode AccuracyOnly

python3 ${MLPERF_INFERENCE}/vision/classification_and_detection/tools/accuracy-coco.py --mlperf-accuracy-file ./mlperf_log_accuracy.json --coco-dir ${IMGDIR} | tee accuracy.txt #--remove-48-empty-images
