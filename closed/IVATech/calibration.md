# Quantization
To perform the neural network acceleration on IVA TPU we need to perform an operation of quantization of network parameters. 

Quantization is a process of conversion of data types with acceptable loss of work quality of networks in order to increase the performance of inference.

## Quantized data types
IVA TPU uses signed INT8 or FP16 for inference in dependence on an operation type. Matrix operations are performed in INT8, vector and element-wise operations are performed in FP16.

## Quantization process
The conversion of the weight coefficients and biases into reduced bits data types is performed via calibration. This process is presented as a selection of scaling coefficients for inputs, outputs and constants of each layer, which are defined on the  basis of input data range. To carry out such a process it is necessary to prepare a numpy array which contains a set of tensors for the input layer from the target dataset.

After the determination of the scaling coefficients we perform the construction of a quantized model for inference on IVA TPU.
