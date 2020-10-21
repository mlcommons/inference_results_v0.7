#!/bin/bash
# This script configures the GPU number #GPU to 7 MIG slices with profile ID 19

GPU=${1:-0}

MIG_ENABLED=$(nvidia-smi --query-gpu=mig.mode.current --format=noheader,csv -i ${GPU})
if [ $MIG_ENABLED == "Enabled" ]
then
echo "Partitioning GPU #$GPU to 7 MIG slices"
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -i $GPU
sudo nvidia-smi mig -cci -i $GPU
else
echo "MIG is not enabled. Please enable MIG on $GPU and rerun"
fi
