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

import os, sys
sys.path.insert(0, os.getcwd())

import json
import platform
import subprocess
import sys

from glob import glob

VERSION = "v0.7"

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s")

from code.common.system_list import system_list

def is_xavier():
    return platform.processor() == "aarch64"

def get_system_id():
    arch = platform.processor()

    if is_xavier():
        # The only officially support aarch64 platform is Jetson Xavier
        with open("/sys/firmware/devicetree/base/model") as product_f:
            product_name = product_f.read()
        if "jetson" in product_name.lower():
            if "AGX" in product_name:
                return "AGX_Xavier"
            elif "NX" in product_name:
                return "Xavier_NX"
            else:
                raise RuntimeError("Unrecognized aarch64 device. Only AGX Xavier and Xavier NX are supported.")

    try:
        import pycuda.driver
        import pycuda.autoinit
        name = pycuda.driver.Device(0).name()
        count_actual = pycuda.driver.Device.count()
    except:
        nvidia_smi_out = run_command("nvidia-smi -L", get_output=True, tee=False)

        # Strip empty lines
        tmp = [ line for line in nvidia_smi_out if len(line) > 0 ]
        count_actual = len(tmp)
        if count_actual == 0:
            raise RuntimeError("nvidia-smi did not detect any GPUs:\n{:}".format(nvidia_smi_out))

        # Format: GPU #: <name> (UUID: <uuid>)
        name = tmp[0].split("(")[0].split(": ")[1].strip()

    system_id, matched, closest = ("", "", -1000)
    for system in system_list:
        if system[1] not in name:
            continue

        # Match exact name with higher priority than partial name
        if matched == name and system[1] != name:
            continue

        closer = (abs(count_actual - system[2]) < abs(count_actual - closest))
        if closer or (matched != name and system[1] == name):
            system_id, matched, closest = system

    if closest == -1000:
        raise RuntimeError("Cannot find valid configs for {:d}x {:}. Please pass in config path using --configs=<PATH>.".format(count_actual, name))
    elif closest != count_actual:
        logging.warn("Cannot find valid configs for {:d}x {:}. Using {:d}x {:} configs instead.".format(count_actual, name, closest, name))
    return system_id

class BENCHMARKS:
    # Official names for benchmarks
    BERT = "bert"
    ALL = [BERT]

    # Whatever we might call it
    alias_map = {
        "BERT": BERT,
        "bert": BERT
    }

    def alias(name):
        if not name in BENCHMARKS.alias_map:
            raise ValueError("Unknown benchmark: {:}".format(name))
        return BENCHMARKS.alias_map[name]

class SCENARIOS:
    # Official names for scenarios
    Offline = "Offline"
    ALL = [Offline]

    # Whatever we might call it
    alias_map = {
        "Offline": Offline,
        "offline": Offline
    }

    def alias(name):
        if not name in SCENARIOS.alias_map:
            raise ValueError("Unknown scenario: {:}".format(name))
        return SCENARIOS.alias_map[name]

def run_command(cmd, get_output=False, tee=True, custom_env=None):
    """
    Runs a command.

    Args:
        cmd (str): The command to run.
        get_output (bool): If true, run_command will return the stdout output. Default: False.
        tee (bool): If true, captures output (if get_output is true) as well as prints output to stdout. Otherwise, does
            not print to stdout.
    """
    logging.info("Running command: {:}".format(cmd))
    if not get_output:
        return subprocess.check_call(cmd, shell=True)
    else:
        output = []
        if custom_env is not None:
            logging.info("Overriding Environment")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, env=custom_env)
        else:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        for line in iter(p.stdout.readline, b""):
            line = line.decode("utf-8")
            if tee:
                sys.stdout.write(line)
                sys.stdout.flush()
            output.append(line.rstrip("\n"))
        ret = p.wait()
        if ret == 0:
            return output
        else:
            raise subprocess.CalledProcessError(ret, cmd)

def args_to_string(d, blacklist=[], delimit=True, double_delimit=False):
    flags = []
    for flag in d:
        # Skip unset
        if d[flag] is None:
            continue
        # Skip blacklisted
        if flag in blacklist:
            continue
        if type(d[flag]) is bool:
            if d[flag] is True:
                flags.append("--{:}=true".format(flag))
            elif d[flag] is False:
                flags.append("--{:}=false".format(flag))
        elif type(d[flag]) in [int, float] or not delimit:
            flags.append("--{:}={:}".format(flag, d[flag]))
        else:
            if double_delimit:
                flags.append("--{:}=\\\"{:}\\\"".format(flag, d[flag]))
            else:
                flags.append("--{:}=\"{:}\"".format(flag, d[flag]))
    return " ".join(flags)

def flags_bool_to_int(d):
    for flag in d:
        if type(d[flag]) is bool:
            if d[flag]:
                d[flag] = 1
            else:
                d[flag] = 0
    return d

def dict_get(d, key, default=None):
    val = d.get(key, default)
    return default if val is None else val

def find_config_files(benchmarks, scenarios):
    config_file_candidates = ["configs/{:}/{:}/config.json".format(benchmark, scenario)
        for scenario in scenarios
        for benchmark in benchmarks
    ]

    # Only return existing files
    config_file_candidates = [i for i in config_file_candidates if os.path.exists(i)]
    return ",".join(config_file_candidates)

def load_configs(config_files):
    configs = []
    for config in config_files.split(","):
        file_locs = glob(config)
        if len(file_locs) == 0:
            raise ValueError("Config file {:} cannot be found.".format(config))
        for file_loc in file_locs:
            with open(file_loc) as f:
                logging.info("Parsing config file {:} ...".format(file_loc))
                configs.append(json.load(f))
    return configs
