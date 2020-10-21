#!/bin/bash
# This script finds a random number n, between 0 and number of MIG devices and sets the CUDA_VISIBLE_DEVICES environment variable to
# the instance 'n'
number_of_instances=$(nvidia-smi -L | grep MIG | wc -l)
random_instance=$(( ( $RANDOM % ${number_of_instances} ) + 1))
echo "Found ${number_of_instances} MIG instances"
instance=$(nvidia-smi -L | grep MIG | cut -d ' ' -f8 | sed 's/)//g' | head -${random_instance} | tail -1)
export CUDA_VISIBLE_DEVICES=${instance}
echo "Setting CUDA_VISIBLE_DEVICES to ${CUDA_VISIBLE_DEVICES}"
