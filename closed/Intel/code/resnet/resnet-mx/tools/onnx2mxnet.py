"""
Converts mlperf ONNX model to mxnet model
"""
import argparse, os
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.contrib.onnx as onnx_mxnet
from mxnet.test_utils import download
from onnx.numpy_helper import to_array


# CLI
def parse_args():
  parser = argparse.ArgumentParser(description='onnx model converter')
  parser.add_argument('--model-path', type=str, default='model',
                      help='onnx model path for conversion, and the output path')
  args = parser.parse_args()
  return args


def download_model(model_name, model_path):
  # reference: https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection
  if model_name == 'resnet50-v1.5':
    model_url = 'https://zenodo.org/record/2592612/files/resnet50_v1.onnx'
    data_shape = (1, 3, 224, 224)
  else:
    raise ValueError('Model: {} not implemented.'.format(model_name))

  if not os.path.exists(model_path):
    os.mkdir(model_path)
  onnx_model_file = os.path.join(model_path, model_name + '.onnx')
  print (onnx_model_file)

  if not os.path.exists(onnx_model_file):
    print("Downloading ONNX model from: {}".format(model_url))
    download(model_url, onnx_model_file)

  return onnx_model_file, data_shape


def _parse_array(tensor_proto):
    """Grab data in TensorProto and convert to numpy array."""
    np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
    return nd.array(np_array)


def _convert_resnet50_v1_5(model_name, model_path, onnx_model_file, data_shape):
  import onnx
  import sys
  sys.path.append("./")
  from resnetv1b import resnet50_v1b
  from map_params import map_from_mxnet_to_onnx

  net = resnet50_v1b(classes=1001)
  onnx_model = onnx.load(onnx_model_file)
  params_onnx = {}
  for onnx_tensor in onnx_model.graph.initializer:
    onnx_tensor_name = onnx_tensor.name
    onnx_tensor_params = _parse_array(onnx_tensor)
    # convert dense weight and bias from class_num=1001 to 1000
    if 'dense/kernel' in onnx_tensor_name:
      onnx_tensor_params = mx.nd.transpose(onnx_tensor_params)
    params_onnx[onnx_tensor_name] = onnx_tensor_params

  net.hybridize()
  data = mx.sym.var('data')
  out = net(data)
  out = mx.sym.SoftmaxOutput(data=out, name='softmax')
  symnet = mx.symbol.load_json(out.tojson())

  net_params = net.collect_params()
  param_key = list(net_params.keys())
  args = {}
  auxs = {}
  map_dict = map_from_mxnet_to_onnx(model_name)
  for k in param_key:
    if k in map_dict:
      onnx_k = map_dict[k]
      if 'running' in k:
          auxs[k] = params_onnx[onnx_k]
      else:
          args[k] = params_onnx[onnx_k]

  mod = mx.mod.Module(symbol=symnet, context=mx.cpu(),
                      label_names = ['softmax_label'])
  mod.bind(for_training=False,
            data_shapes=[('data', data_shape)])
  mod.set_params(arg_params=args, aux_params=auxs)
  save_prefix = os.path.join(model_path, 'resnet50_v1b')
  mod.save_checkpoint(save_prefix, 0)


def convert_model_from_onnx(model_name, model_path, test_image=None):
  onnx_model_file, data_shape = download_model(model_name, model_path)
  if model_name == 'resnet50-v1.5':
    _convert_resnet50_v1_5(model_name, model_path, onnx_model_file, data_shape)
  else:
    raise ValueError('Model: {} not implemented.'.format(model_name))


if __name__ == '__main__':
  args = parse_args()
  model_name = 'resnet50-v1.5'
  model_path = args.model_path

  print("Convert ONNX to MXNet")
  convert_model_from_onnx(model_name, model_path)
  print("Done, Save to {}".format(model_name))
