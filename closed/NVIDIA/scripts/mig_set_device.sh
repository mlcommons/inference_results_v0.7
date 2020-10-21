#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script finds a random number n, between 0 and number of MIG devices and sets the CUDA_VISIBLE_DEVICES environment variable to
# the instance 'n'
number_of_instances=$(nvidia-smi -L | grep MIG | wc -l)
random_instance=$(( ( $RANDOM % ${number_of_instances} ) + 1))
echo "Found ${number_of_instances} MIG instances"
instance=$(nvidia-smi -L | grep MIG | cut -d ' ' -f8 | sed 's/)//g' | head -${random_instance} | tail -1)
export CUDA_VISIBLE_DEVICES=${instance}
echo "Setting CUDA_VISIBLE_DEVICES to ${CUDA_VISIBLE_DEVICES}"
