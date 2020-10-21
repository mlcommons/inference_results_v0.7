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

import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
from code.common import BENCHMARKS, SCENARIOS

arguments_dict = {
    # Common arguments
    "gpu_batch_size": {
        "help": "GPU batch size to use for the engine.",
        "type": int,
    },
    "dla_batch_size": {
        "help": "DLA batch size to use for the engine.",
        "type": int,
    },
    "batch_size": {
        "help": "Batch size to use for the engine.",
        "type": int,
    },
    "verbose": {
        "help": "Use verbose output",
        "action": "store_true",
    },
    "verbose_nvtx": {
        "help": "Turn ProfilingVerbosity to kVERBOSE so that layer detail is printed in NVTX.",
        "action": "store_true",
    },
    "no_child_process": {
        "help": "Do not generate engines on child process. Do it on current process instead.",
        "action": "store_true"
    },
    "workspace_size": {
        "help": "The maximum size of temporary workspace that any layer in the network can use in TRT",
        "type": int,
        "default": None
    },

    # Power measurements
    "power": {
        "help": "Select if you would like to measure power",
        "action": "store_true"
    },

    # Dataset location
    "data_dir": {
        "help": "Directory containing unprocessed datasets",
        "default": os.environ.get("DATA_DIR", "build/data"),
    },
    "preprocessed_data_dir": {
        "help": "Directory containing preprocessed datasets",
        "default": os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"),
    },

    # Arguments related to precision
    "precision": {
        "help": "Precision. Default: int8",
        "choices": ["fp32", "fp16", "int8", None],
        # None needs to be set as default since passthrough arguments will
        # contain a default value and override configs. Specifying None as the
        # default will cause it to not be inserted into passthrough / override
        # arguments.
        "default": None,
    },
    "input_dtype": {
        "help": "Input datatype. Choices: fp32, int8.",
        "choices": ["fp32", "fp16", "int8", None],
        "default": None
    },
    "input_format": {
        "help": "Input format (layout). Choices: linear, chw4",
        "choices": ["linear", "chw4", "dhwc8", None],
        "default": None
    },
    "audio_fp16_input": {
        "help": "Is input format for raw audio file in fp16?. Choices: true, false",
        "action": "store_true",
    },

    # Arguments related to quantization calibration
    "force_calibration": {
        "help": "Run quantization calibration even if the cache exists. (Only used for quantized models)",
        "action": "store_true",
    },
    "calib_batch_size": {
        "help": "Batch size for calibration.",
        "type": int
    },
    "calib_max_batches": {
        "help": "Number of batches to run for calibration.",
        "type": int
    },
    "cache_file": {
        "help": "Path to calibration cache.",
        "default": None,
    },
    "calib_data_map": {
        "help": "Path to the data map of the calibration set.",
        "default": None,
    },

    # Benchmark configuration arguments
    "scenario": {
        "help": "Name for the scenario. Used to generate engine name.",
    },
    "dla_core": {
        "help": "DLA core to use. Do not set if not using DLA",
        "default": None,
    },
    "model_path": {
        "help": "Path to the model (weights) file.",
    },
    "active_sms": {
        "help": "Control the percentage of active SMs while generating engines.",
        "type": int
    },

    # Profiler selection
    "profile": {
        "help": "[INTERNAL ONLY] Select if you would like to profile -- select among nsys, nvprof and ncu",
        "type": str
    },

    # Harness configuration arguments
    "log_dir": {
        "help": "Directory for all output logs.",
        "default": os.environ.get("LOG_DIR", "build/logs/default"),
    },
    "use_graphs": {
        "help": "Enable CUDA graphs",
        "action": "store_true",
    },

    # RNN-T Harness
    "nopipelined_execution": {
        "help": "Disable pipelined execution",
        "action": "store_true",
    },

    "nobatch_sorting": {
        "help": "Disable batch sorting by sequence length",
        "action": "store_true",
    },

    "noenable_audio_processing": {
        "help": "Disable DALI preprocessing and fallback to preprocessed npy files",
        "action": "store_true",
    },

    "nouse_copy_kernel": {
        "help": "Disable using DALI's scatter gather kernel instead of using cudamemcpyAsync",
        "action": "store_true",
    },

    "num_warmups": {
        "help": "Number of samples to warmup on. A value of -1 runs two full batches for each stream (2*batch_size*streams_per_gpu*NUM_GPUS), 0 turns off warmups.",
        "type": int
    },

    "max_seq_length": {
        "help": "Max sequence length for audio",
        "type": int
    },

    "audio_batch_size": {
        "help": "Batch size for DALI's processing",
        "type": int
    },

    "audio_buffer_num_lines": {
        "help": "Number of audio samples in flight for DALI's processing",
        "type": int
    },

    "dali_batches_issue_ahead": {
        "help": "Number of batches for which cudamemcpy is issued ahead of DALI compute",
        "type": int
    },

    "dali_pipeline_depth": {
        "help": "Depth of sub-batch processing in DALI pipeline",
        "type": int
    },

    # LWIS settings
    "devices": {
        "help": "Comma-separated list of numbered devices",
    },
    "map_path": {
        "help": "Path to map file for samples",
    },
    "tensor_path": {
        "help": "Path to preprocessed samples in .npy format",
    },
    "performance_sample_count": {
        "help": "Number of samples to load in performance set.  0=use default",
        "type": int,
    },
    "gpu_copy_streams": {
        "help": "Number of copy streams to use for GPU",
        "type": int,
    },
    "gpu_inference_streams": {
        "help": "Number of inference streams to use for GPU",
        "type": int,
    },
    "dla_copy_streams": {
        "help": "Number of copy streams to use for DLA",
        "type": int,
    },
    "dla_inference_streams": {
        "help": "Number of inference streams to use for DLA",
        "type": int,
    },
    "run_infer_on_copy_streams": {
        "help": "Run inference on copy streams.",
    },
    "warmup_duration": {
        "help": "Minimum duration to perform warmup for",
        "type": float,
    },
    "use_direct_host_access": {
        "help": "Use direct access to host memory for all devices",
    },
    "use_deque_limit": {
        "help": "Use a max number of elements dequed from work queue",
    },
    "deque_timeout_us": {
        "help": "Timeout in us for deque from work queue.",
        "type": int,
    },
    "use_batcher_thread_per_device": {
        "help": "Enable a separate batcher thread per device",
    },
    "use_cuda_thread_per_device": {
        "help": "Enable a separate cuda thread per device",
    },
    "start_from_device": {
        "help": "Assuming that inputs start from device memory in QSL"
    },
    "max_dlas": {
        "help": "Max number of DLAs to use per device",
        "type": int,
    },
    "coalesced_tensor": {
        "help": "Turn on if all the samples are coalesced into one single npy file"
    },
    "assume_contiguous": {
        "help": "Assume that the data in a query is already contiguous"
    },
    "complete_threads": {
        "help": "Number of threads per device for sending responses",
        "type": int,
    },
    "use_same_context": {
        "help": "Use the same TRT context for all copy streams (shape must be static and gpu_inference_streams must be 1).",
    },

    # Shared settings
    "mlperf_conf_path": {
        "help": "Path to mlperf.conf",
    },
    "user_conf_path": {
        "help": "Path to user.conf",
    },

    # Loadgen settings
    "test_mode": {
        "help": "Testing mode for Loadgen",
        "choices": ["SubmissionRun", "AccuracyOnly", "PerformanceOnly", "FindPeakPerformance"],
    },
    "min_duration": {
        "help": "Minimum test duration",
        "type": int,
    },
    "max_duration": {
        "help": "Maximum test duration",
        "type": int,
    },
    "min_query_count": {
        "help": "Minimum number of queries in test",
        "type": int,
    },
    "max_query_count": {
        "help": "Maximum number of queries in test",
        "type": int,
    },
    "qsl_rng_seed": {
        "help": "Seed for RNG that specifies which QSL samples are chosen for performance set and the order in which samples are processed in AccuracyOnly mode",
        "type": int,
    },
    "sample_index_rng_seed": {
        "help": "Seed for RNG that specifies order in which samples from performance set are included in queries",
        "type": int,
    },

    # Loadgen logging settings
    "logfile_suffix": {
        "help": "Specify the filename suffix for the LoadGen log files",
    },
    "logfile_prefix_with_datetime": {
        "help": "Prefix filenames for LoadGen log files",
        "action": "store_true",
    },
    "log_copy_detail_to_stdout": {
        "help": "Copy LoadGen detailed logging to stdout",
        "action": "store_true",
    },
    "disable_log_copy_summary_to_stdout": {
        "help": "Disable copy LoadGen summary logging to stdout",
        "action": "store_true",
    },
    "log_mode": {
        "help": "Logging mode for Loadgen",
        "choices": ["AsyncPoll", "EndOfTestOnly", "Synchronous"],
    },
    "log_mode_async_poll_interval_ms": {
        "help": "Specify the poll interval for asynchrounous logging",
        "type": int,
    },
    "log_enable_trace": {
        "help": "Enable trace logging",
    },

    # Triton args
    "use_triton": {
        "help": "Use Triton harness",
        "action": "store_true",
    },
    "preferred_batch_size": {
        "help": "Preferred batch sizes"
    },
    "max_queue_delay_usec": {
        "help": "Set max queuing delay in usec.",
        "type": int
    },
    "instance_group_count": {
        "help": "Set number of instance groups on each GPU.",
        "type": int
    },
    "request_timeout_usec": {
        "help": "Set the timeout for every request in usec.",
        "type": int
    },

    # Server harness arguments
    "server_target_qps": {
        "help": "Target QPS for server scenario.",
        "type": int,
    },
    "server_target_latency_ns": {
        "help": "Desired latency constraint for server scenario",
        "type": int,
    },
    "server_target_latency_percentile": {
        "help": "Desired latency percentile constraint for server scenario",
        "type": float,
    },
    # not supported by current Loadgen - when support is added use the Loadgen default
    # "server_coalesce_queries": {
    #    "help": "Enable coalescing outstanding queries in the server scenario",
    #    "action": "store_true",
    # },
    "schedule_rng_seed": {
        "help": "Seed for RNG that affects the poisson arrival process in server scenario",
        "type": int,
    },
    "accuracy_log_rng_seed": {
        "help": "Affects which samples have their query returns logged to the accuracy log in performance mode.",
        "type": int,
    },

    # Single stream harness arguments
    "single_stream_expected_latency_ns": {
        "help": "Inverse of desired target QPS",
        "type": int,
    },
    "single_stream_target_latency_percentile": {
        "help": "Desired latency percentile for single stream scenario",
        "type": float,
    },

    # Offline harness arguments
    "offline_expected_qps": {
        "help": "Target samples per second rate for the SUT",
        "type": float,
    },

    # Multi stream harness arguments
    "multi_stream_target_qps": {
        "help": "Target QPS rate for the SUT",
        "type": float,
    },
    "multi_stream_target_latency_ns": {
        "help": "Desired latency constraint for multi stream scenario",
        "type": int,
    },
    "multi_stream_target_latency_percentile": {
        "help": "Desired latency percentile for multi stream scenario",
        "type": float,
    },
    "multi_stream_samples_per_query": {
        "help": "Expected samples per query for multi stream scenario",
        "type": int,
    },
    "multi_stream_max_async_queries": {
        "help": "Max number of asynchronous queries for multi stream scenario",
        "type": int,
    },

    # Args used by code.main
    "action": {
        "help": "generate_engines / run_harness / calibrate / generate_conf_files",
        "choices": ["generate_engines", "run_harness", "calibrate", "generate_conf_files", "run_audit_harness", "run_audit_verification"],
    },
    "benchmarks": {
        "help": "Specify the benchmark(s) with a comma-separated list. " +
        "Default: run all benchmarks.",
        "default": None,
    },
    "configs": {
        "help": "Specify the config files with a comma-separated list. " +
            "Wild card (*) is also allowed. If \"\", detect platform and attempt to load configs. " +
            "Default: \"\"",
        "default": "",
    },
    "config_ver": {
        "help": "Config version to run. Uses 'default' if not set.",
        "default": "default",
    },
    "scenarios": {
        "help": "Specify the scenarios with a comma-separated list. " +
            "Choices:[\"Server\", \"Offline\", \"SingleStream\", \"MultiStream\"] " +
            "Default: \"*\"",
        "default": None,
    },
    "no_gpu": {
        "help": "Do not perform action with GPU parameters (run on DLA only).",
        "action": "store_true",
    },
    "gpu_only": {
        "help": "Only perform action with GPU parameters (do not run DLA).",
        "action": "store_true",
    },
    "audit_test": {
        "help": "Defines audit test to run.",
        "choices": ["TEST01", "TEST04-A", "TEST04-B", "TEST05"],
    },
    "system_name": {
        "help": "Override the system name to run under",
        "type": str
    },

    # Args used for engine runners
    "engine_file": {
        "help": "File to load engine from",
    },
    "num_samples": {
        "help": "Number of samples to use for accuracy runner",
        "type": int,
    },

    # DLRM harness
    "sample_partition_path": {
        "help": "Path to sample partition file in npy format.",
    },
    "num_staging_threads": {
        "help": "Number of staging threads in DLRM BatchMaker",
        "type": int,
    },
    "num_staging_batches": {
        "help": "Number of staging batches in DLRM BatchMaker",
        "type": int,
    },
    "max_pairs_per_staging_thread": {
        "help": "Maximum pairs to copy in one BatchMaker staging thread",
        "type": int,
    },
    "gpu_num_bundles": {
        "help": "Number of event+buffer bundles per GPU (default: 2)",
        "type": int,
        "default": 2,
    },
    "check_contiguity": {
        "help": "Check if inputs are already contiguous in QSL to avoid copying",
        "action": "store_true",
    },
    "use_jemalloc":
    {
        "help": "Use libjemalloc.so.2 as the malloc(3) implementation",
        "action": "store_true",
    },
    "numa_config": {
        "help": "NUMA settings: cpu cores for each GPU, assuming each GPU corresponds to one NUMA node",
    },
}

# ================== Argument groups ================== #

# Engine generation
PRECISION_ARGS = ["precision", "input_dtype", "input_format", "audio_fp16_input"]
CALIBRATION_ARGS = ["verbose", "force_calibration", "calib_batch_size", "calib_max_batches", "cache_file",
                    "calib_data_map", "model_path"]
GENERATE_ENGINE_ARGS = ["dla_core", "gpu_batch_size", "dla_batch_size", "gpu_copy_streams", "workspace_size",
                        "gpu_inference_streams", "verbose_nvtx", "max_seq_length"] + PRECISION_ARGS + CALIBRATION_ARGS

# Harness framework arguments
LOADGEN_ARGS = ["test_mode", "min_duration", "max_duration", "min_query_count",
    "max_query_count", "qsl_rng_seed", "sample_index_rng_seed", "schedule_rng_seed", "accuracy_log_rng_seed", "logfile_suffix",
    "logfile_prefix_with_datetime", "log_copy_detail_to_stdout", "disable_log_copy_summary_to_stdout",
    "log_mode", "log_mode_async_poll_interval_ms", "log_enable_trace",
    "single_stream_target_latency_percentile", "multi_stream_target_latency_percentile",
    "multi_stream_target_qps", "multi_stream_target_latency_ns", "multi_stream_max_async_queries",
    "server_target_latency_percentile", "server_target_qps", "server_target_latency_ns" ]
LWIS_ARGS = ["devices", "gpu_copy_streams", "gpu_inference_streams",
    "dla_batch_size", "dla_copy_streams", "dla_inference_streams", "run_infer_on_copy_streams", "warmup_duration", "use_direct_host_access", "use_deque_limit", "deque_timeout_us",
    "use_batcher_thread_per_device", "use_cuda_thread_per_device", "start_from_device", "max_dlas", "coalesced_tensor", "assume_contiguous", "complete_threads", "use_same_context"]
TRITON_ARGS = ["instance_group_count", "preferred_batch_size", "max_queue_delay_usec", "request_timeout_usec"]
SHARED_ARGS = [ "use_graphs", "gpu_batch_size", "map_path", "tensor_path", "performance_sample_count", "mlperf_conf_path", "user_conf_path" ]
OTHER_HARNESS_ARGS = ["log_dir", "use_triton", "sample_partition_path", "nopipelined_execution", "nobatch_sorting", "noenable_audio_processing", "audio_batch_size", "num_staging_threads",
                      "num_staging_batches", "max_pairs_per_staging_thread", "gpu_num_bundles", "check_contiguity", "use_jemalloc",
                      "nouse_copy_kernel", "num_warmups", "dali_batches_issue_ahead", "dali_pipeline_depth", "audio_buffer_num_lines", "max_seq_length"]
HARNESS_ARGS = ["verbose", "scenario"] + PRECISION_ARGS + LOADGEN_ARGS + LWIS_ARGS + TRITON_ARGS + SHARED_ARGS + OTHER_HARNESS_ARGS

# Scenario dependent arguments. These are prefixed with device: "gpu_", "dla_", "concurrent_"
OFFLINE_PARAMS = ["offline_expected_qps"]
SINGLE_STREAM_PARAMS = ["single_stream_expected_latency_ns"]
MULTI_STREAM_PARAMS = ["multi_stream_samples_per_query"]
SERVER_PARAMS = []

# Wrapper for scenario+harness
OFFLINE_HARNESS_ARGS = OFFLINE_PARAMS + HARNESS_ARGS
SINGLE_STREAM_HARNESS_ARGS = SINGLE_STREAM_PARAMS + HARNESS_ARGS
MULTI_STREAM_HARNESS_ARGS = MULTI_STREAM_PARAMS + HARNESS_ARGS
# LOADGEN_ARGS + OTHER_HARNESS_ARGS + ["gpu_batch_size", "map_path", "tensor_path"]
SERVER_HARNESS_ARGS = SERVER_PARAMS + HARNESS_ARGS

# For code.main
MAIN_ARGS = [
    "action",
    "benchmarks",
    "configs",
    "config_ver",
    "scenarios",
    "no_gpu",
    "gpu_only",
    "profile",
    "no_child_process",
    "power",
    "audit_test",
    "system_name"
]

# For accuracy runners
ACCURACY_ARGS = ["verbose", "engine_file", "batch_size", "num_samples"]


def parse_args(whitelist):
    parser = argparse.ArgumentParser()
    for flag in whitelist:
        if flag not in arguments_dict:
            raise IndexError("Unknown flag '{:}'".format(flag))

        parser.add_argument("--{:}".format(flag), **arguments_dict[flag])
    return vars(parser.parse_known_args()[0])


def check_args():
    parser = argparse.ArgumentParser()
    for flag in arguments_dict:
        parser.add_argument("--{:}".format(flag), **arguments_dict[flag])
    parser.parse_args()

##
# @brief Create an argument list based on scenario and benchmark name


def getScenarioBasedHarnessArgs(scenario, benchmark):
    arglist = None
    if scenario == SCENARIOS.SingleStream:
        arglist = SINGLE_STREAM_HARNESS_ARGS
    elif scenario == SCENARIOS.Offline:
        arglist = OFFLINE_HARNESS_ARGS
    elif scenario == SCENARIOS.MultiStream:
        arglist = MULTI_STREAM_HARNESS_ARGS
    elif scenario == SCENARIOS.Server:
        arglist = SERVER_HARNESS_ARGS
    else:
        raise RuntimeError("Unknown Scenario \"{}\"".format(scenario))

    return arglist
