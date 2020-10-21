# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import os
import logging
import mxnet as mx
from mxnet import nd, image
from mxnet.contrib.quantization import *
from functools import partial
import ctypes


def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at %s' % fname)
    sym.save(fname)


def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k): v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLPerf Resnet50-v1b quantization script')
    parser.add_argument('--model', type=str, default='resnet50_v1',
                        help='model to be quantized.')
    parser.add_argument('--epoch', type=int, default=0,
                        help='number of epochs, default is 0')
    parser.add_argument('--model-path', type=str, default='./model',
                         help='model directory.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--dataset-list", required=True, help="path to the dataset list")
    parser.add_argument("--calib-dataset-file", required=True, help="path to the calib dataset list file")
    parser.add_argument("--cache", type=int, default=0, help="use cache")
    parser.add_argument("--cache-dir", type=str, default=None, help="cache directory")
    parser.add_argument("--count", type=int, help="number of dataset items to use")
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--num-calib-batches', type=int, default=10,
                        help='number of batches for calibration')
    parser.add_argument("--num-phy-cpus", default=56, type=int, help="number of pyhsical cpu cores")
    parser.add_argument('--exclude-first-conv', action='store_true', default=False,
                        help='excluding quantizing the first conv layer since the'
                             ' input data may have negative value which doesn\'t support at moment' )
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the calibration dataset')
    parser.add_argument('--ilit-config', type=str, default='./config.yaml')
    parser.add_argument('--calib-mode', type=str, default='entropy',
                        help='calibration mode used for generating calibration table for the quantized symbol; supports'
                             ' 1. none: no calibration will be used. The thresholds for quantization will be calculated'
                             ' on the fly. This will result in inference speed slowdown and loss of accuracy'
                             ' in general.'
                             ' 2. naive: simply take min and max values of layer outputs as thresholds for'
                             ' quantization. In general, the inference accuracy worsens with more examples used in'
                             ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                             ' inference results.'
                             ' 3. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                             ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                             ' kinds of quantized models if the calibration dataset is representative enough of the'
                             ' inference dataset.')
    parser.add_argument('--quantized-dtype', type=str, default='auto',
                        choices=['auto', 'int8', 'uint8'],
                        help='quantization destination data type for input data')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='suppress most of log')
    args = parser.parse_args()
    ctx = mx.cpu(0)
    logger = None
    if not args.quiet:
        logging.basicConfig()
        logger = logging.getLogger('quantize_model')
        logger.setLevel(logging.INFO)

    calib_mode = args.calib_mode
    batch_size = args.batch_size
    # get number of batches for calibration
    num_calib_batches = args.num_calib_batches
    # get image shape
    image_shape = args.image_shape
    label_name = args.label_name
    data_shape = tuple([int(i) for i in image_shape.split(',')])

    if logger:
        logger.info(args)
        logger.info('shuffle_dataset={}'.format(args.shuffle_dataset))
        logger.info('calibration mode set to {}'.format(calib_mode))
        logger.info('batch size = {} for calibration'.format(batch_size))
        logger.info('label_name = {}'.format(label_name))
        logger.info('Input data shape = {}'.format(data_shape))

        if calib_mode == 'none':
            logger.info('skip calibration step as calib_mode is none')
        else:
            logger.info('number of batches = {} for calibration'.format(num_calib_batches))

    model_path = args.model_path
    prefix = os.path.join(model_path, args.model)

    epoch = args.epoch

    # load FP32 model
    fp32_model = mx.model.load_checkpoint(prefix, epoch)

    import sys
    sys.path.append('python/')
    import imagenet
    import dataset
    import re
    pre_proc = dataset.pre_process_vgg
    logger.info('preprocess with validation dataset with {} cpus, it will take a while (depends '
                'on how many cpus are used) to convert the raw image to numpy ndarray for the very '
                'first time'.format(
                    args.num_phy_cpus))
    ds = imagenet.Imagenet(data_path=args.dataset_path,
                           image_list=args.dataset_list,
                           name='imagenet',
                           image_format='NCHW',
                           pre_process=dataset.pre_process_vgg,
                           use_cache=args.cache,
                           cache_dir=args.cache_dir,
                           use_int8=False,
                           num_workers=args.num_phy_cpus,
                           count=args.count)

    # get the calibration sample index
    image_index_list = []
    with open(args.calib_dataset_file, 'r') as f:
        for s in f:
            image_name = re.split(r"\s+", s.strip())[0]
            index = ds.image_list.index(image_name)
            image_index_list.append(index)
    assert len(image_index_list) >= (batch_size * num_calib_batches), \
        'the samples specified ({}x{}) for calibration is more than provided ({})'.format(
            batch_size, num_calib_batches, len(image_index_list))

    ds.load_query_samples(image_index_list)
    data, label = ds.get_samples(image_index_list)
    calib_data = mx.io.NDArrayIter(data=data,
                                   label=label,
                                   batch_size=args.batch_size,
                                   shuffle=args.shuffle_dataset)

    from ilit import Tuner
    cnn_tuner = Tuner(args.ilit_config)
    ilit_model = cnn_tuner.tune(fp32_model, q_dataloader=calib_data, eval_dataloader=calib_data)

    qsym, qarg_params, aux_params = ilit_model

    suffix = '-quantized'
    sym_name = '%s-symbol.json' % (prefix + suffix)
    save_symbol(sym_name, qsym, logger)
    param_name = '%s-%04d.params' % (prefix + suffix, epoch)
    save_params(param_name, qarg_params, aux_params, logger)

    # collect min and max range for the calibration dataset
    data_min = 100
    data_max = -100
    for elem in data:
        elem_min = elem.min()
        elem_max = elem.max()
        if elem_min < data_min:
            data_min = elem_min
        if elem_max > data_max:
            data_max = elem_max

    ds.scale_factor = float(127) / max(abs(data_min), abs(data_max))
    logger.info('calibration dataset min={}, max={}, scale_factor={}'.format(data_min, data_max, ds.scale_factor))

    def mxnet_quantize_wrapper(arr, data_min, data_max):
        mx_arr = mx.nd.array(arr)
        quantized_val, _, _ = mx.nd.contrib.quantize(data=mx_arr,
                                                     min_range=mx.nd.array(data_min),
                                                     max_range=mx.nd.array(data_max),
                                                     out_type='int8')
        return quantized_val.asnumpy()

    mxnet_quantize_callback = partial(mxnet_quantize_wrapper,
                                     data_min=data_min,
                                     data_max=data_max)
    ds.quantize_dataset(mxnet_quantize_callback)
