# MLPerf Inference v0.7 - Calibration

## For GPU results

Please refer to NVIDIA's calibration.md file for documentations about quantization process and weight transformations.

## For CPU rsults

Please refer to Intel's calibration for documentation about quantization process and weight transformations


## For FPGA Zebra results

The network parameters and the activations are quantized to 8-bit signed integers. The quantization is performed by multiplying the parameters by scaling values, clamping the results to the 8-bit integer range and rounding to nearest integer. No retraining is performed nor labeled data used.
The Argmax is not quantized and executed on the CPU.

The quantization uses a small subset of images from one of the MLPerf calibration sets, and is performed in about 1 minute. The scaling parameters are chosen using the data obtained from computing the subset.
The weights are quantized using per-channel symmetric quantization.
The biases and activations are quantized using per-tensor symmetric quantization.

## For FPGA Xilinx results

We used per-tensor symmetric for both weights and activation and uses an 8-bit integer as its numerical precision.
Moreover, a power-of-2 scaling factor is used in quantization. 
The formula is: Q(x) = clamp(round(x * scale), -128, 127). Here scale = 2 ^ p and p is an integer value.
In post-training quantization, the distributions of weights and activation are used to get the optimal scaling factors.

### Weights

A per-tensor symmetric quantization is used. The scaling factor of each tensor is obtained according to the distribution by minimizing the mean square error of the quantized and float tensor.
Both weight tensors and bias tensors are quantized to int8.

### Activations

A per-tensor symmetric quantization is used. The scaling factor of each tensor is calibrated by invoking the model on the calibration dataset (from the mlperf calibration dataset). The histogram of the scaling factor is record over mini-batches and the most commonly used value is set.
Based on the scaling factor the activation tensor is clamped and quantized.  

### Furthur improvement

To improve quantization performance we employ cross layer equalization[1] when needed.

### Quantization in Plugins

Xilinx's closed division submissions use our proprietary software stack named Vitis-AI[2], which implements the scheme described above. 

### Open Division

Our Open Division submissions use exactly the same calibration and quantization setting. 

### References
[1]Nagel, Markus, et al. "Data-free quantization through weight equalization and bias correction." Proceedings of the IEEE International Conference on Computer Vision. 2019.<br />
[2]Vitis-AI User Guide, https://www.xilinx.com/support/documentation/sw_manuals/vitis_ai/1_2/ug1414-vitis-ai.pdf, 2020
