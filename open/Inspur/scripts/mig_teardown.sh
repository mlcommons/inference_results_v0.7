#!/bin/bash

GPU=${1:-0}

echo "Destroying GPU #$GPU partitions"
sudo nvidia-smi mig -dci -i $GPU
sudo nvidia-smi mig -dgi -i $GPU
