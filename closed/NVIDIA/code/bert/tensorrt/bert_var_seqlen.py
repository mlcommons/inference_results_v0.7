#!/usr/bin/env python3
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

#TODO for now, this is loads the precompiled dev versions of the BERT TRT plugins from yko's TRT fork
import pycuda
import pycuda.autoinit
import tensorrt as trt
import os, sys
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common import logging, dict_get, BENCHMARKS
from code.common.builder import BenchmarkBuilder
from code.bert.tensorrt.builder_utils import BertConfig, load_onnx_fake_quant
from code.bert.tensorrt.int8_builder_var_seqlen import bert_squad_int8_var_seqlen
from code.bert.tensorrt.int8_builder_vs_il import bert_squad_int8_vs_il
from code.bert.tensorrt.fp16_builder_var_seqlen import bert_squad_fp16_var_seqlen

#to run with a different seq_len, we need to run preprocessing again and point to the resulting folder
# by setting the variable:
#PREPROCESSED_DATA_DIR=/data/projects/bert/squad/v1.1/s128_q64_d128/

# to build engines in lwis mode, we expect a single sequence length and a single batch size
class BERTBuilder(BenchmarkBuilder):

    def __init__(self, args):
        workspace_size = dict_get(args, "workspace_size", default=(5 << 30))
        logging.info("Use workspace_size: {:}".format(workspace_size))
        super().__init__(args, name=BENCHMARKS.BERT, workspace_size=workspace_size)
        self.bert_config_path = "code/bert/tensorrt/bert_config.json"

        self.seq_len = 384 # default sequence length

        self.batch_size = dict_get(args, "batch_size", default=1)

        self.num_profiles = 1
        if 'gpu_inference_streams' in args:
            # use gpu_inference_streams to determine the number of duplicated profiles
            # in the engine when not using lwis mode
            self.num_profiles = args['gpu_inference_streams']

        self.is_int8 = args['precision'] == 'int8'

        if self.is_int8:
            self.model_path = dict_get(args, "model_path", default="build/models/bert/bert_large_v1_1_fake_quant.onnx")
        else:
            self.model_path = dict_get(args, "model_path", default="build/models/bert/bert_large_v1_1.onnx")

        self.bert_config = BertConfig(self.bert_config_path)

        self.enable_il = False
        if self.is_int8 and 'enable_interleaved' in args:
            self.enable_il = args['enable_interleaved']

        if self.batch_size > 512: 
            # tactics selection is limited at very large batch sizes
            self.builder_config.max_workspace_size = 7 << 30
        if 'nx' in self.system_id.lower():
            # use 1GB only for XavierNX
            self.builder_config.max_workspace_size = 1 << 30
        

    def initialize(self):
        self.initialized = True

    def _get_engine_name(self, device_type, batch_size):
        if device_type is None:
            device_type = self.device_type

        return "{:}/{:}-{:}-{:}-{:}_S_{:}_B_{:}_P_{:}_vs{:}.{:}.plan".format(
            self.engine_dir, self.name, self.scenario,
            device_type, self.precision, self.seq_len, self.batch_size, self.num_profiles,'_il' if self.enable_il else '', self.config_ver)

    """
    Calls self.initialize() if it has not been called yet.
    Creates optimization profiles for multiple SeqLen and BatchSize combinations
    Builds and saves the engine.
    TODO do we also need multiple profiles per setting?
    """
    def build_engines(self):

        # Load weights
        weights_dict = load_onnx_fake_quant(self.model_path)

        if not self.initialized:
            self.initialize()

        # Create output directory if it does not exist.
        if not os.path.exists(self.engine_dir):
            os.makedirs(self.engine_dir)

        input_shape = (-1, )
        cu_seqlen_shape = (-1,)

        self.profiles = []

        with self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network:

            # Looks like the tactics available with even large WS are not competitive anyway.
            # Might be able to reduce this also

            self.builder_config.set_flag(trt.BuilderFlag.FP16)
            if self.is_int8:
                self.builder_config.set_flag(trt.BuilderFlag.INT8)
                if self.enable_il:
                    bert_squad_int8_vs_il(network, weights_dict, self.bert_config, input_shape, cu_seqlen_shape)
                else:
                    bert_squad_int8_var_seqlen(network, weights_dict, self.bert_config, input_shape, cu_seqlen_shape)
            else:
                bert_squad_fp16_var_seqlen(network, weights_dict, self.bert_config, input_shape, cu_seqlen_shape)

            engine_name = self._get_engine_name(self.device_type, None)
            logging.info("Building {:}".format(engine_name))

            # The harness expectss i -> S -> B. This should be fine, since now there is only one S per engine
            for i in range(self.num_profiles):
                profile = self.builder.create_optimization_profile()
                assert network.num_inputs == 4, "Unexpected number of inputs"
                assert network.get_input(0).name == 'input_ids'
                assert network.get_input(1).name == 'segment_ids'
                assert network.get_input(2).name == 'cu_seqlens'
                assert network.get_input(3).name == 'max_seqlen'

                B = self.batch_size
                S = self.seq_len
                # TODO Like this, we can only control granularity using multiples of max_seqlen (B*S)
                # Investigate if this can be improved otherwise
                min_shape = (1,) # TODO is it an issue to cover such a wide range?
                max_shape = (B*S,)
                profile.set_shape('input_ids', min_shape, max_shape, max_shape)
                profile.set_shape('segment_ids', min_shape, max_shape, max_shape)
                profile.set_shape('cu_seqlens', (1+1,), (B+1,), (B+1,))
                profile.set_shape('max_seqlen', (1,), (S,), (S,))
                if not profile:
                    raise RuntimeError("Invalid optimization profile!")
                self.builder_config.add_optimization_profile(profile)
                self.profiles.append(profile)

            # Build engines
            engine = self.builder.build_engine(network, self.builder_config)
            assert engine is not None, "Engine Build Failed!"
            buf = engine.serialize()
            with open(engine_name, 'wb') as f:
                f.write(buf)

    # BERT does not need calibration.
    def calibrate(self):
        logging.info("BERT does not need calibration.")
