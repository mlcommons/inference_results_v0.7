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

import json
import re
import traceback
import platform

import os, sys
sys.path.insert(0, os.getcwd())

from code.common import logging, get_system_id
from code.common.scopedMPS import turn_off_mps
from code.common import args_to_string, find_config_files, load_configs, run_command
from code.common import BENCHMARKS, SCENARIOS
import code.common.arguments as common_args
from importlib import import_module
import multiprocessing as mp
import time

def get_benchmark(conf):
    benchmark_name = conf["benchmark"]

    if benchmark_name == BENCHMARKS.BERT:
        # TODO now only BERT uses gpu_inference_streams to generate engines
        conf = apply_overrides(conf, ['gpu_inference_streams'])
        BERTBuilder = import_module("code.bert.tensorrt_sparse.bert_var_seqlen").BERTBuilder
        return BERTBuilder(conf)
    else:
        raise ValueError("Unknown benchmark: {:}".format(benchmark_name))

def apply_overrides(config, keys):
    # Make a copy so we don't modify original dict
    config = dict(config)
    override_args = common_args.parse_args(keys)
    for key in override_args:
        # Unset values (None) and unset store_true values (False) are both false-y
        if override_args[key]:
            config[key] = override_args[key]
    return config

def flatten_config(config, system_id):
    benchmark_conf = config.get(system_id, None)
    if benchmark_conf is not None:
        # Passthrough for top level values
        benchmark_conf["system_id"] = system_id
        benchmark_conf["scenario"] = config["scenario"]
        benchmark_conf["benchmark"] = config["benchmark"]
    return benchmark_conf

def handle_generate_engine(config, gpu=True, dla=True, copy_from_default=False):
    benchmark_name = config["benchmark"]

    logging.info("Building engines for {:} benchmark in {:} scenario...".format(benchmark_name, config["scenario"]))

    start_time = time.time()

    arglist = common_args.GENERATE_ENGINE_ARGS
    config = apply_overrides(config, arglist)

    if gpu and "gpu_batch_size" in config:
        config["batch_size"] = config["gpu_batch_size"]
        config["dla_core"] = None
        logging.info("Building GPU engine for {:}_{:}_{:}".format(config["system_id"], benchmark_name, config["scenario"]))
        b = get_benchmark(config)

        if copy_from_default:
            copy_default_engine(b)
        else:
            b.build_engines()

    end_time = time.time()

    logging.info("Finished building engines for {:} benchmark in {:} scenario.".format(benchmark_name, config["scenario"]))

    print("Time taken to generate engines: {:} seconds".format(end_time - start_time))

def handle_run_harness(config, gpu=True, dla=True, profile=None, power=False, generate_conf_files_only=False):
    benchmark_name = config["benchmark"]

    logging.info("Running harness for {:} benchmark in {:} scenario...".format(benchmark_name, config["scenario"]))

    arglist = common_args.getScenarioBasedHarnessArgs(config["scenario"], benchmark_name)

    config = apply_overrides(config, arglist)

    # Validate arguments

    if not dla:
        config["dla_batch_size"] = None
    if not gpu:
        config["gpu_batch_size"] = None

    # If we only want to generate conf_files, then set flag to true
    if generate_conf_files_only:
        config["generate_conf_files_only"] = True
        profile = None
        power = False

    if benchmark_name == BENCHMARKS.BERT:
        from code.bert.tensorrt_sparse.harness import BertHarness
        harness = BertHarness(config, name=benchmark_name)
        config["inference_server"] = "custom"
    else:
        raise RuntimeError("Could not find supported harness!")

    # Attempt to run profiler. Note that this is only available internally.
    if profile is not None:
        try:
            from code.internal.profiler import ProfilerHarness
            harness = ProfilerHarness(harness, profile)
        except:
            logging.info("Could not load profiler: Are you an internal user?")

    if power:
        try:
            from code.internal.power_measurements import PowerMeasurements
            power_measurements = PowerMeasurements("{}/{}/{}".format(
                os.getcwd(),
                "power_measurements",
                config.get("config_name"))
            )
            power_measurements.start()
        except:
            power_measurements = None

    for key, value in config.items():
        print("{} : {}".format(key, value))
    result = ""

    # Launch the harness
    passed = True
    try:
        result = harness.run_harness()
        logging.info("Result: {:}".format(result))
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        passed = False
    finally:
        if power and power_measurements is not None:
            power_measurements.stop()
    if not passed:
        raise RuntimeError("Run harness failed!")

    if generate_conf_files_only and result == "Generated conf files":
        return

    # Append result to perf result summary log.
    log_dir = config["log_dir"]
    summary_file = os.path.join(log_dir, "perf_harness_summary.json")
    results = {}
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            results = json.load(f)

    config_name = "{:}-{:}-{:}".format(harness.get_system_name(), config["config_ver"], config["scenario"])
    if config_name not in results:
        results[config_name] = {}
    results[config_name][benchmark_name] = result

    with open(summary_file, "w") as f:
        json.dump(results, f)

    # Check accuracy from loadgen logs.
    accuracy = check_accuracy(os.path.join(harness.get_full_log_dir(), "mlperf_log_accuracy.json"), config)
    summary_file = os.path.join(log_dir, "accuracy_summary.json")
    results = {}
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            results = json.load(f)

    if config_name not in results:
        results[config_name] = {}
    results[config_name][benchmark_name] = accuracy

    with open(summary_file, "w") as f:
        json.dump(results, f)

def check_accuracy(log_file, config):
    benchmark_name = config["benchmark"]

    accuracy_targets = {
        BENCHMARKS.BERT: 90.874,
    }
    threshold_ratio = float(config["accuracy_level"][:-1]) / 100

    if not os.path.exists(log_file):
        return "Cannot find accuracy JSON file."
    with open(log_file, "r") as f:
        loadgen_dump = json.load(f)
    if len(loadgen_dump) == 0:
        return "No accuracy results in PerformanceOnly mode."

    dtype_expand_map = {"fp16": "float16", "fp32": "float32", "int8": "float16"} # Use FP16 output for INT8 mode

    threshold = accuracy_targets[benchmark_name] * threshold_ratio
    if benchmark_name == BENCHMARKS.BERT:
        dtype = config["precision"].lower()
        if dtype in dtype_expand_map:
            dtype = dtype_expand_map[dtype]
        val_data_path = os.path.join(os.environ.get("DATA_DIR", "build/data"), "squad", "dev-v1.1.json")
        vocab_file_path = "build/models/bert/vocab.txt"
        output_prediction_path = os.path.join(os.path.dirname(log_file), "predictions.json")
        cmd = "python3 build/inference/language/bert/accuracy-squad.py " \
            "--log_file {:} --vocab_file {:} --val_data {:} --out_file {:} " \
            "--output_dtype {:}".format(log_file, vocab_file_path, val_data_path,
                    output_prediction_path, dtype)
        regex = r".*\"f1\": ([0-9\.]+)"
    else:
        raise ValueError("Unknown benchmark: {:}".format(benchmark_name))

    output = run_command(cmd, get_output=True)
    result_regex = re.compile(regex)
    accuracy = None
    with open(os.path.join(os.path.dirname(log_file), "accuracy.txt"), "w") as f:
        for line in output:
            print(line, file=f)
    for line in output:
        result_match = result_regex.match(line)
        if not result_match is None:
            accuracy = float(result_match.group(1))
            break

    accuracy_result = "PASSED" if accuracy is not None and accuracy >= threshold else "FAILED"

    if accuracy_result == "FAILED":
        raise RuntimeError("Accuracy = {:.3f}, Threshold = {:.3f}. Accuracy test {:}!".format(accuracy, threshold, accuracy_result))

    return "Accuracy = {:.3f}, Threshold = {:.3f}. Accuracy test {:}.".format(accuracy, threshold, accuracy_result)

def main(main_args, system_id):
    # Turn off MPS in case it's turned on.
    turn_off_mps()

    benchmarks = BENCHMARKS.ALL
    if main_args["benchmarks"] is not None:
        benchmarks = main_args["benchmarks"].split(",")
        for i, benchmark in enumerate(benchmarks):
            benchmarks[i] = BENCHMARKS.alias(benchmark)
    scenarios = SCENARIOS.ALL
    if main_args["scenarios"] is not None:
        scenarios = main_args["scenarios"].split(",")
        for i, scenario in enumerate(scenarios):
            scenarios[i] = SCENARIOS.alias(scenario)

    profile = main_args.get("profile", None)
    power = main_args.get("power", False)

    # Automatically detect architecture and scenarios and load configs
    config_files = main_args["configs"]
    if config_files == "" or config_files is None:
        config_files = find_config_files(benchmarks, scenarios)
        if config_files == "":
            logging.warn("Cannot find any valid configs for the specified benchmark-scenario pairs.")
            return

    logging.info("Using config files: {:}".format(str(config_files)))
    configs = load_configs(config_files)

    for config in configs:
        base_benchmark_conf = flatten_config(config, system_id)
        if base_benchmark_conf is None:
            continue

        base_benchmark_conf["config_name"] = "{:}_{:}_{:}".format(
            system_id,
            base_benchmark_conf["benchmark"],
            base_benchmark_conf["scenario"]
        )
        logging.info("Processing config \"{:}\"".format(base_benchmark_conf["config_name"]))

        # Load config_ver / apply overrides
        conf_vers = main_args.get("config_ver", "default").split(",")

        # Build default first. This is because some config_vers only modify harness args, and the engine is the same as
        # default. In this case, we build default first, and copy it instead of rebuilding it.
        if "default" in conf_vers:
            conf_vers = ["default"] + list(set(conf_vers) - {"default"})
        elif "all" in conf_vers:
            conf_vers = ["default"] + list(base_benchmark_conf.get("config_ver", {}).keys())

        for conf_ver in conf_vers:
            benchmark_conf = dict(base_benchmark_conf) # Copy the config so we don't modify it

            # These fields are canonical names that refer to certain config versions
            benchmark_conf["accuracy_level"] = "99%"
            benchmark_conf["optimization_level"] = "plugin-enabled"
            benchmark_conf["inference_server"] = "lwis"

            """@etcheng
            NOTE: The original plan was to use a syntax like high_accuracy+triton to be able to combine already defined
            config_vers. However, since high_accuracy, triton, and high_accuracy+triton are likely to all have different
            expected QPS values, it makes more sense to keep high_accuracy_triton as a separate, individual config_ver.

            In the future, perhaps we can make an "extends": [ list of strings ] or { dict of config_ver name ->
            config_key } field in config_vers, so that we can define new config_vers that extend or combine previous
            config_vers.
            """

            equiv_to_default = False

            if conf_ver != "default":
                if "config_ver" not in benchmark_conf or conf_ver not in benchmark_conf["config_ver"]:
                    logging.warn("--config_ver={:} does not exist in config file '{:}'".format(conf_ver, benchmark_conf["config_name"]))
                    continue
                else:
                    if "high_accuracy" in conf_ver:
                        benchmark_conf["accuracy_level"] = "99.9%"
                    if "ootb" in conf_ver:
                        benchmark_conf["optimization_level"] = "ootb"
                    # "inference_server" is set when we run the harness

                    overrides = benchmark_conf["config_ver"][conf_ver]

                    # Check if this config_ver is equivalent to the default engine
                    gen_eng_argset = set(common_args.GENERATE_ENGINE_ARGS)
                    override_argset = set(overrides.keys())
                    equiv_to_default = (len(gen_eng_argset & override_argset) == 0)

                    benchmark_conf.update(overrides)

            # Update the config_ver key to be the actual string name, not the overrides
            benchmark_conf["config_ver"] = conf_ver

            need_gpu = not main_args["no_gpu"]
            need_dla = not main_args["gpu_only"]

            if main_args["action"] == "generate_engines":
                # Turn on MPS if server scenario and if active_sms is specified.
                benchmark_conf = apply_overrides(benchmark_conf, ["active_sms"])
                active_sms = benchmark_conf.get("active_sms", None)

                copy_from_default = ("default" in conf_vers) and equiv_to_default
                if copy_from_default:
                    logging.info("config_ver={:} only modifies harness args. Re-using default engine.".format(conf_ver))

                _gen_args = [ benchmark_conf ]
                _gen_kwargs = {
                    "gpu": need_gpu,
                    "dla": need_dla,
                    "copy_from_default": copy_from_default
                }

                handle_generate_engine(*_gen_args, **_gen_kwargs)
            elif main_args["action"] == "run_harness":
                handle_run_harness(benchmark_conf, need_gpu, need_dla, profile, power)
            elif main_args["action"] == "generate_conf_files":
                handle_run_harness(benchmark_conf, need_gpu, need_dla, generate_conf_files_only=True)

if __name__ == "__main__":
    mp.set_start_method("spawn")

    # Check any invalid/misspelling flags.
    common_args.check_args()
    main_args = common_args.parse_args(common_args.MAIN_ARGS)

    # Load System ID
    system_id = get_system_id()
    logging.info("Detected System ID: " + system_id)

    main(main_args, system_id)
