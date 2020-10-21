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

LOADGEN_PATH ?= ../../../build/inference/loadgen
LOADGEN_INC=$(LOADGEN_PATH)
LOADGEN_FLAGS= $(LOADGEN_PATH)/build/libmlperf_loadgen.a

BUILD_DIR ?= ../../../code/resnet50/int4
NVCC  ?= $(CUDA)/bin/nvcc

CUDNN_FLAGS = -L/usr/lib/x86-64-linux-gnu -I/usr/include -lcudnn
CUBLAS_FLAGS = -lcublas

.PHONY: dirs all clean

OBJS = $(BUILD_DIR)/int4_offline.a

all: dirs int4_offline

dirs:
	if [ ! -d $(BUILD_DIR) ] ; then mkdir $(BUILD_DIR)    ; fi

clean:
	rm -f int4_offline

int4_offline:
	$(NVCC) -o $@ $(OBJS) $(LOADGEN_FLAGS) $(CUDNN_FLAGS) $(CUBLAS_FLAGS)

.PHONY: build_int4
build_int4:
ifeq ($(ARCH), x86_64)
	@echo "Building INT4 harness..."
	cd $(INT4_DIR) \
		&& make -j CUDA=$(CUDA_PATH) BUILD_DIR=$(INT4_DIR) LOADGEN_PATH=$(LOADGEN_INCLUDE_DIR) clean  \
		&& make -j CUDA=$(CUDA_PATH) BUILD_DIR=$(INT4_DIR) LOADGEN_PATH=$(LOADGEN_INCLUDE_DIR) all
endif

###############################  RUN  ###############################

.PHONY: run_intx_test_harness
run_intx_test_harness: link_dataset_dir
	@mkdir -p $(LOG_DIR)/ResNet50_int4
	@cd $(INT4_DIR) && ./int4_offline $(INT4_ARGS) --lgls_logfile_outdir $(LOG_DIR)/ResNet50_int4
	@if [ `stat -c "%s" $(LOG_DIR)/ResNet50_int4/mlperf_log_accuracy.json` -gt 4 ]; then \
		python3 $(INFERENCE_DIR)/vision/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file $(LOG_DIR)/ResNet50_int4/mlperf_log_accuracy.json --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32 ; \
	fi

.PHONY: run_int4_TitanRTXx4_accuracy
run_int4_TitanRTXx4_accuracy: link_dataset_dir
	@mkdir -p $(LOG_DIR)/TitanRTXx4/resnet/Offline/accuracy
	@cd $(INT4_DIR) && ./int4_offline -b 1024 -a autoconfig_rtx --mlperf_conf_path $(PROJECT_ROOT)/measurements/TitanRTXx4/resnet/Offline/mlperf.conf --user_conf_path $(PROJECT_ROOT)/measurements/TitanRTXx4/resnet/Offline/user.conf --test-mode AccuracyOnly --lgls_logfile_outdir $(LOG_DIR)/TitanRTXx4/resnet/Offline/accuracy
	@python3 $(INFERENCE_DIR)/vision/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file $(LOG_DIR)/TitanRTXx4/resnet/Offline/accuracy/mlperf_log_accuracy.json --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32 2>&1 | tee $(LOG_DIR)/TitanRTXx4/resnet/Offline/accuracy/accuracy.txt

.PHONY: run_int4_TitanRTXx4_performance
run_int4_TitanRTXx4_performance: link_dataset_dir
	@mkdir -p $(LOG_DIR)/TitanRTXx4/resnet/Offline/performance/run_1
	@cd $(INT4_DIR) && ./int4_offline -b 1024 -a autoconfig_rtx --mlperf_conf_path $(PROJECT_ROOT)/measurements/TitanRTXx4/resnet/Offline/mlperf.conf --user_conf_path $(PROJECT_ROOT)/measurements/TitanRTXx4/resnet/Offline/user.conf --test-mode PerformanceOnly --lgls_logfile_outdir $(LOG_DIR)/TitanRTXx4/resnet/Offline/performance/run_1

.PHONY: run_int4_T4x8_accuracy
run_int4_T4x8_accuracy: link_dataset_dir
	@mkdir -p $(LOG_DIR)/T4x8/resnet/Offline/accuracy
	@cd $(INT4_DIR) && ./int4_offline -b 512 -a autoconfig_t4 --mlperf_conf_path $(PROJECT_ROOT)/measurements/T4x8/resnet/Offline/mlperf.conf --user_conf_path $(PROJECT_ROOT)/measurements/T4x8/resnet/Offline/user.conf --test-mode AccuracyOnly --lgls_logfile_outdir $(LOG_DIR)/T4x8/resnet/Offline/accuracy
	@python3 $(INFERENCE_DIR)/vision/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file $(LOG_DIR)/T4x8/resnet/Offline/accuracy/mlperf_log_accuracy.json --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32 2>&1 | tee $(LOG_DIR)/T4x8/resnet/Offline/accuracy/accuracy.txt

.PHONY: run_int4_T4x8_performance
run_int4_T4x8_performance: link_dataset_dir
	@mkdir -p $(LOG_DIR)/T4x8/resnet/Offline/performance/run_1
	@cd $(INT4_DIR) && ./int4_offline -b 512 -a autoconfig_t4 --mlperf_conf_path $(PROJECT_ROOT)/measurements/T4x8/resnet/Offline/mlperf.conf --user_conf_path $(PROJECT_ROOT)/measurements/T4x8/resnet/Offline/user.conf --test-mode PerformanceOnly --lgls_logfile_outdir $(LOG_DIR)/T4x8/resnet/Offline/performance/run_1

.PHONY: run_int4_T4x20_accuracy
run_int4_T4x20_accuracy: link_dataset_dir
	@mkdir -p $(LOG_DIR)/T4x20/resnet/Offline/accuracy
	@cd $(INT4_DIR) && ./int4_offline -b 512 -a autoconfig_t4 --mlperf_conf_path $(PROJECT_ROOT)/measurements/T4x20/resnet/Offline/mlperf.conf --user_conf_path $(PROJECT_ROOT)/measurements/T4x20/resnet/Offline/user.conf --test-mode AccuracyOnly --lgls_logfile_outdir $(LOG_DIR)/T4x20/resnet/Offline/accuracy
	@python3 $(INFERENCE_DIR)/vision/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file $(LOG_DIR)/T4x20/resnet/Offline/accuracy/mlperf_log_accuracy.json --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32 2>&1 | tee $(LOG_DIR)/T4x20/resnet/Offline/accuracy/accuracy.txt

.PHONY: run_int4_T4x20_performance
run_int4_T4x20_performance: link_dataset_dir
	@mkdir -p $(LOG_DIR)/T4x20/resnet/Offline/performance/run_1
	@cd $(INT4_DIR) && ./int4_offline -b 512 -a autoconfig_t4 --mlperf_conf_path $(PROJECT_ROOT)/measurements/T4x20/resnet/Offline/mlperf.conf --user_conf_path $(PROJECT_ROOT)/measurements/T4x20/resnet/Offline/user.conf --test-mode PerformanceOnly --lgls_logfile_outdir $(LOG_DIR)/T4x20/resnet/Offline/performance/run_1