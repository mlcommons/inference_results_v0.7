#!/bin/bash
set -ex

export OMP_NUM_THREADS=28
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

if [ "x$DATASET_PATH" == "x" ]; then
    echo "DATASET_PATH not set" && exit 1
fi
if [ "x$DATASET_LIST" == "x" ]; then
    echo "DATASET_LIST not set" && exit 1
fi
if [ "x$CALIBRATION_IMAGE_LIST" == "x" ]; then
    echo "CALIBRATION_IMAGE_LIST not set" && exit 1
fi

# Need to update dataset-path/dataset-list/cache-dir/calib-dataset-file
# according to your local env
python tools/quantize_model.py \
    --model-path=./model \
    --model=resnet50_v1b \
    --batch-size=1 \
    --num-calib-batches=500 \
    --calib-mode=entropy \
    --ilit-config=./tools/cnn.yaml \
    --dataset-path=$DATASET_PATH \
    --dataset-list=$DATASET_LIST \
    --cache=1 \
    --calib-dataset-file=$CALIBRATION_IMAGE_LIST
