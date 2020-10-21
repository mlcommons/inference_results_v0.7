# MLPerf Inference v0.7 - Calibration

## OpenVINO results

Please refer to Intel's v0.7 calibration document.

## TensorRT results

Please refer to NVIDIA's [v0.5 calibration document](https://github.com/mlperf/inference_results_v0.5/blob/master/closed/NVIDIA/calibration.md).

## TFLite/ArmNN results

We use quantized TFLite models shared by Google. We believe that Google used [post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) via the [TFLite converter](https://www.tensorflow.org/lite/convert/).

## RNN-T results

We use [dynamic quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html) for the PyTorch LSTM plugins.
