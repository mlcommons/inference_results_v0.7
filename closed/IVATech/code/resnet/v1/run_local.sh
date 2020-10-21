#!/bin/bash

set -x
name="resnet50-iva-fpga-1"
common_opt="--config ../mlperf.conf"
dataset="--dataset $DATA_DIR"
OUTPUT_DIR=`pwd`/output/$name # default output dir
MLPERF_CONF=`pwd`/mlperf.conf
USER_CONF=`pwd`/user.conf

while getopts "o:" opt
do
	case $opt in
	o) echo "Output dir $OPTARG"
		OUTPUT_DIR=$OPTARG
		;;
	*) echo "Nothing found";;
	esac
done

shift "$((OPTIND-1))"

MODEL=$1
SCENARIO=$2
MODE=$3

if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
cd $OUTPUT_DIR

iva_$MODEL $SCENARIO $MODE $dataset -c $MLPERF_CONF -u $USER_CONF -v ~/datasets/imagenet/val_map.txt -m $MODEL_DIR/$MODEL.tpu