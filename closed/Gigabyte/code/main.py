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

import os
import sys
sys.path.insert(0, os.getcwd())

from code.common import logging, get_system_id, is_xavier
from code.common.scopedMPS import ScopedMPS, turn_off_mps
from code.common import args_to_string, find_config_files, load_configs, run_command
from code.common import BENCHMARKS, SCENARIOS
from code.common import auditing
import code.common.arguments as common_args
from importlib import import_module
import multiprocessing as mp
from multiprocessing import Process
import time
import shutil

def get_benchmark(conf):
    benchmark_name = conf["benchmark"]

    # Do not use a map. We want to import benchmarks as we need them, because some take
    # time to load due to plugins.
    if benchmark_name == BENCHMARKS.ResNet50:
        ResNet50 = import_module("code.resnet50.tensorrt.ResNet50").ResNet50
        return ResNet50(conf)
    elif benchmark_name == BENCHMARKS.SSDResNet34:
        SSDResNet34 = import_module("code.ssd-resnet34.tensorrt.SSDResNet34").SSDResNet34
        return SSDResNet34(conf)
    elif benchmark_name == BENCHMARKS.SSDMobileNet:
        SSDMobileNet = import_module("code.ssd-mobilenet.tensorrt.SSDMobileNet").SSDMobileNet
        return SSDMobileNet(conf)
    elif benchmark_name == BENCHMARKS.BERT:
        # TODO now only BERT uses gpu_inference_streams to generate engines
        conf = apply_overrides(conf, ['gpu_inference_streams'])
        BERTBuilder = import_module("code.bert.tensorrt.bert_var_seqlen").BERTBuilder
        return BERTBuilder(conf)
    elif benchmark_name == BENCHMARKS.RNNT:
        RNNTBuilder = import_module("code.rnnt.tensorrt.rnn-t_builder").RNNTBuilder
        return RNNTBuilder(conf)
    elif benchmark_name == BENCHMARKS.DLRM:
        DLRMBuilder = import_module("code.dlrm.tensorrt.dlrm").DLRMBuilder
        return DLRMBuilder(conf)
    elif benchmark_name == BENCHMARKS.UNET:
        UNETBuilder = import_module("code.3d-unet.tensorrt.3d-unet").UnetBuilder
        return UNETBuilder(conf)
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

def launch_handle_generate_engine(*args, **kwargs):
    retries = 1
    timeout = 7200
    success = False
    for i in range(retries):
        # Build engines in another process to make sure we exit with clean cuda
        # context so that MPS can be turned off.
        from code.main import handle_generate_engine
        p = Process(target=handle_generate_engine, args=args, kwargs=kwargs)
        p.start()
        try:
            p.join(timeout)
        except KeyboardInterrupt:
            p.terminate()
            p.join(timeout)
            raise KeyboardInterrupt
        if p.exitcode == 0:
            success = True
            break

    if not success:
        raise RuntimeError("Building engines failed!")

def copy_default_engine(benchmark):
    new_path = benchmark._get_engine_name(None, None)  # Use default values
    benchmark.config_ver = "default"
    default_path = benchmark._get_engine_name(None, None)

    logging.info("Copying {:} to {:}".format(default_path, new_path))
    shutil.copyfile(default_path, new_path)

def handle_generate_engine(config, gpu=True, dla=True, copy_from_default=False):
    benchmark_name = config["benchmark"]

    logging.info(
        "Building engines for {:} benchmark in {:} scenario...".format(
            benchmark_name,
            config["scenario"]))

    start_time = time.time()

    arglist = common_args.GENERATE_ENGINE_ARGS
    config = apply_overrides(config, arglist)

    if dla and "dla_batch_size" in config:
        config["batch_size"] = config["dla_batch_size"]
        logging.info("Building DLA engine for {:}_{:}_{:}".format(config["system_id"], benchmark_name, config["scenario"]))
        b = get_benchmark(config)

        if copy_from_default:
            copy_default_engine(b)
        else:
            b.build_engines()

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

def handle_audit_verification(audit_test_name, config):
    # Decouples the verification step from any auditing runs for better maintenance and testing
    logging.info('AUDIT HARNESS: Running verification script...')
    # Prepare log_dir
    config['log_dir'] = os.path.join('build/compliance_logs', audit_test_name)
    # Get a harness object
    harness, config = _generate_harness_object(config=config, profile=None)

    result = None
    if audit_test_name == 'TEST01':
        result = auditing.verify_test01(harness)
    elif audit_test_name == 'TEST04-A' or audit_test_name == 'TEST04-B':
        exclude_list = [BENCHMARKS.BERT, BENCHMARKS.DLRM, BENCHMARKS.RNNT]
        if BENCHMARKS.alias(config['benchmark']) in exclude_list:
            logging.info('TEST04 is not supported for benchmark {}. Ignoring request...'.format(config['benchmark']))
            return None
        result = auditing.verify_test04(harness)
    elif audit_test_name == 'TEST05':
        result = auditing.verify_test05(harness)
    return result

def _generate_harness_object(config, profile):
    # Refactors harness generation for use by functions other than handle_run_harness
    benchmark_name = config['benchmark']
    if config.get("use_triton"):
        from code.common.server_harness import TritonHarness
        harness = TritonHarness(config, name=benchmark_name)
        config["inference_server"] = "triton"
    elif benchmark_name == BENCHMARKS.BERT:
        from code.bert.tensorrt.harness import BertHarness
        harness = BertHarness(config, name=benchmark_name)
        config["inference_server"] = "custom"
    elif benchmark_name == BENCHMARKS.DLRM:
        from code.dlrm.tensorrt.harness import DLRMHarness
        harness = DLRMHarness(config, name=benchmark_name)
        config["inference_server"] = "custom"
    elif benchmark_name == BENCHMARKS.RNNT:
        from code.rnnt.tensorrt.harness import RNNTHarness
        harness = RNNTHarness(config, name=benchmark_name)
        config["inference_server"] = "custom"
    else:
        from code.common.lwis_harness import LWISHarness
        harness = LWISHarness(config, name=benchmark_name)

    # Attempt to run profiler. Note that this is only available internally.
    if profile is not None:
        try:
            from code.internal.profiler import ProfilerHarness
            harness = ProfilerHarness(harness, profile)
        except BaseException:
            logging.info("Could not load profiler: Are you an internal user?")

    return harness, config

def handle_run_harness(config, gpu=True, dla=True, profile=None,
                       power=False, generate_conf_files_only=False, compliance=False):
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

    harness, config = _generate_harness_object(config, profile)

    if power:
        try:
            from code.internal.power_measurements import PowerMeasurements
            power_measurements = PowerMeasurements("{}/{}/{}".format(
                os.getcwd(),
                "power_measurements",
                config.get("config_name"))
            )
            power_measurements.start()
        except BaseException:
            power_measurements = None

    for key, value in config.items():
        print("{} : {}".format(key, value))
    result = ""

    if compliance:
        # AP: We need to keep the compliance logs separated from accuracy and perf
        # otherwise it messes up the update_results process
        config['log_dir'] = os.path.join('build/compliance_logs', config['audit_test_name'])
        logging.info('AUDIT HARNESS: Overriding log_dir for compliance run. Set to ' + config['log_dir'])

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

    config_name = "{:}-{:}-{:}".format(harness.get_system_name(),
                                       config["config_ver"],
                                       config["scenario"])
    if config_name not in results:
        results[config_name] = {}
    results[config_name][benchmark_name] = result

    with open(summary_file, "w") as f:
        json.dump(results, f)

    # Check accuracy from loadgen logs.
    if not compliance:
        # TEST01 fails the accuracy test because it produces fewer predictions than expected
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
        BENCHMARKS.ResNet50: 76.46,
        BENCHMARKS.SSDResNet34: 20.0,
        BENCHMARKS.SSDMobileNet: 22.0,
        BENCHMARKS.BERT: 90.874,
        BENCHMARKS.DLRM: 80.25,
        BENCHMARKS.RNNT: 100.0 - 7.45225,
        BENCHMARKS.UNET: 0.853
    }
    threshold_ratio = float(config["accuracy_level"][:-1]) / 100

    if not os.path.exists(log_file):
        return "Cannot find accuracy JSON file."
    # checking if log_file is empty by just reading first several bytes
    # indeed, first 4B~6B is likely all we need to check: '', '[]', '[]\r', '[\n]\n', '[\r\n]\r\n', ...
    # but checking 8B for safety
    with open(log_file, 'r') as lf:
        first_8B = lf.read(8)
        if not first_8B or ('[' in first_8B and ']' in first_8B):
            return "No accuracy results in PerformanceOnly mode."

    dtype_expand_map = {"fp16": "float16", "fp32": "float32", "int8": "float16"} # Use FP16 output for INT8 mode
    accuracy_regex_map = import_module("build.inference.tools.submission.submission-checker").ACC_PATTERN

    threshold = accuracy_targets[benchmark_name] * threshold_ratio
    if benchmark_name in [BENCHMARKS.ResNet50]:
        cmd = "python3 build/inference/vision/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file {:} \
            --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32 ".format(log_file)
        regex = accuracy_regex_map["acc"]
    elif benchmark_name == BENCHMARKS.SSDResNet34:
        cmd = "python3 build/inference/vision/classification_and_detection/tools/accuracy-coco.py --mlperf-accuracy-file {:} \
            --coco-dir {:} --output-file build/ssd-resnet34-results.json --use-inv-map".format(
            log_file, os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "coco"))
        regex = accuracy_regex_map["mAP"]
    elif benchmark_name == BENCHMARKS.SSDMobileNet:
        cmd = "python3 build/inference/vision/classification_and_detection/tools/accuracy-coco.py --mlperf-accuracy-file {:} \
            --coco-dir {:} --output-file build/ssd-mobilenet-results.json".format(
            log_file, os.path.join(os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"), "coco"))
        regex = accuracy_regex_map["mAP"]
    elif benchmark_name == BENCHMARKS.BERT:
        # Having issue installing tokenizers on Xavier...
        if is_xavier():
            cmd = "python3 code/bert/tensorrt/accuracy-bert.py --mlperf-accuracy-file {:} --squad-val-file {:}".format(
                log_file, os.path.join(os.environ.get("DATA_DIR", "build/data"), "squad", "dev-v1.1.json"))
        else:
            dtype = config["precision"].lower()
            if dtype in dtype_expand_map:
                dtype = dtype_expand_map[dtype]
            val_data_path = os.path.join(
                os.environ.get("DATA_DIR", "build/data"),
                "squad", "dev-v1.1.json")
            vocab_file_path = "build/models/bert/vocab.txt"
            output_prediction_path = os.path.join(os.path.dirname(log_file), "predictions.json")
            cmd = "python3 build/inference/language/bert/accuracy-squad.py " \
                "--log_file {:} --vocab_file {:} --val_data {:} --out_file {:} " \
                "--output_dtype {:}".format(log_file, vocab_file_path, val_data_path, output_prediction_path, dtype)
        regex = accuracy_regex_map["F1"]
    elif benchmark_name == BENCHMARKS.DLRM:
        cmd = "python3 build/inference/recommendation/dlrm/pytorch/tools/accuracy-dlrm.py --mlperf-accuracy-file {:} " \
              "--day-23-file build/data/criteo/day_23 --aggregation-trace-file " \
              "build/preprocessed_data/criteo/full_recalib/sample_partition_trace.txt".format(log_file)
        regex = accuracy_regex_map["AUC"]
    elif benchmark_name == BENCHMARKS.RNNT:
        # Having issue installing librosa on Xavier...
        if is_xavier():
            cmd = "python3 code/rnnt/tensorrt/accuracy.py --loadgen_log {:}".format(log_file)
        else:
            # RNNT output indices are in INT8
            cmd = "python3 build/inference/speech_recognition/rnnt/accuracy_eval.py " \
                "--log_dir {:} --dataset_dir build/preprocessed_data/LibriSpeech/dev-clean-wav " \
                "--manifest build/preprocessed_data/LibriSpeech/dev-clean-wav.json " \
                "--output_dtype int8".format(os.path.dirname(log_file))
        regex = accuracy_regex_map["WER"]
    elif benchmark_name == BENCHMARKS.UNET:
        postprocess_dir = "build/brats_postprocessed_data"
        if not os.path.exists(postprocess_dir):
            os.makedirs(postprocess_dir)
        dtype = config["precision"].lower()
        if dtype in dtype_expand_map:
            dtype = dtype_expand_map[dtype]
        cmd = "python3 build/inference/vision/medical_imaging/3d-unet/accuracy-brats.py --log_file {:} " \
            "--output_dtype {:} --preprocessed_data_dir build/preprocessed_data/brats/brats_reference_preprocessed " \
            "--postprocessed_data_dir {:} " \
            "--label_data_dir build/preprocessed_data/brats/brats_reference_raw/Task043_BraTS2019/labelsTr".format(log_file, dtype, postprocess_dir)
        regex = accuracy_regex_map["DICE"]
        # Having issue installing nnUnet on Xavier...
        if is_xavier():
            logging.warning(
                "Accuracy checking for 3DUnet is not supported on Xavier. Please run the following command on desktop:\n{:}".format(cmd))
            cmd = 'echo "Accuracy: mean = 1.0000, whole tumor = 1.0000, tumor core = 1.0000, enhancing tumor = 1.0000"'
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
        raise RuntimeError(
            "Accuracy = {:.3f}, Threshold = {:.3f}. Accuracy test {:}!".format(
                accuracy, threshold, accuracy_result))

    return "Accuracy = {:.3f}, Threshold = {:.3f}. Accuracy test {:}.".format(
        accuracy, threshold, accuracy_result)

def handle_calibrate(config):
    benchmark_name = config["benchmark"]

    logging.info("Generating calibration cache for Benchmark \"{:}\"".format(benchmark_name))
    config = apply_overrides(config, common_args.CALIBRATION_ARGS)
    config["dla_core"] = None
    config["force_calibration"] = True
    b = get_benchmark(config)
    b.calibrate()

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
            benchmark_conf = dict(base_benchmark_conf)  # Copy the config so we don't modify it

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
                    logging.warn(
                        "--config_ver={:} does not exist in config file '{:}'".format(conf_ver, benchmark_conf["config_name"]))
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

            # Override the system_name if it exists
            if "system_name" in main_args:
                benchmark_conf["system_name"] = main_args["system_name"]

            if main_args["action"] == "generate_engines":
                # Turn on MPS if server scenario and if active_sms is specified.
                benchmark_conf = apply_overrides(benchmark_conf, ["active_sms"])
                active_sms = benchmark_conf.get("active_sms", None)

                copy_from_default = ("default" in conf_vers) and equiv_to_default
                if copy_from_default:
                    logging.info(
                        "config_ver={:} only modifies harness args. Re-using default engine.".format(conf_ver))

                _gen_args = [benchmark_conf]
                _gen_kwargs = {
                    "gpu": need_gpu,
                    "dla": need_dla,
                    "copy_from_default": copy_from_default
                }

                if not main_args["no_child_process"]:
                    if config["scenario"] == SCENARIOS.Server and active_sms is not None and active_sms < 100:
                        with ScopedMPS(active_sms):
                            launch_handle_generate_engine(*_gen_args, **_gen_kwargs)
                    else:
                        launch_handle_generate_engine(*_gen_args, **_gen_kwargs)
                else:
                    handle_generate_engine(*_gen_args, **_gen_kwargs)
            elif main_args["action"] == "run_harness":
                # In case there's a leftover audit.config file from a prior compliance run or other reason
                # we need to delete it or we risk silent failure.
                auditing.cleanup()

                handle_run_harness(benchmark_conf, need_gpu, need_dla, profile, power)
            elif main_args["action"] == "run_audit_harness":
                logging.info('\n\n\nRunning compliance harness for test ' + main_args['audit_test'] + '\n\n\n')

                # Find the correct audit.config file and move it in current directory
                dest_config = auditing.load(main_args['audit_test'], benchmark_conf['benchmark'])

                # Make sure the log_file override is valid
                os.makedirs("build/compliance_logs", exist_ok=True)

                # Pass audit test name to handle_run_harness via benchmark_conf
                benchmark_conf['audit_test_name'] = main_args['audit_test']

                # Run harness
                handle_run_harness(benchmark_conf, need_gpu, need_dla, profile, power, compliance=True)

                # Cleanup audit.config
                logging.info("AUDIT HARNESS: Cleaning Up audit.config...")
                auditing.cleanup()
            elif main_args["action"] == "run_audit_verification":
                logging.info("Running compliance verification for test " + main_args['audit_test'])
                handle_audit_verification(audit_test_name=main_args['audit_test'], config=benchmark_conf)
                auditing.cleanup()
            elif main_args["action"] == "calibrate":
                # To generate calibration cache, we only need to run each benchmark once.
                # Use offline config.
                if benchmark_conf["scenario"] == SCENARIOS.Offline:
                    handle_calibrate(benchmark_conf)
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
