# Deci.AI MLPerf Submission 
​
Our model was generated and optimized using four key stages:
1. Deploy Deci's hardware-aware NAS to find the best model that can preserve the desired accuracy.
2. Re-Train the model using standard training techniques. 
3. Compile the model using Intel's OpenVino graph compiler.
4. Apply accuracy-aware 8bit quantization. 
​
## NAS
​
We define the AutoNAC search space for the NAS by leveraging SotA concepts/insights on effective architectures,
The process advances so as to maximize an accuracy-latency (or throughput) trade-off utility function.
 Accuracy is straightforwardly measured (and maintained) and runtime performance measurements are taken on the target HW. 
The latter is done using DIPS (Deci Inference Performance Simulator) as a micro-service for benchmarking on many hardwares.
​
## Graph compilation
​
Intel provides the open-source OpenVINO toolkit.
By installing OpenVINO on the target hardware we leverage the power of the leading CPU graph compiler.
- Conversion - we follow Intel's 'model-optimizer' tool with the script: `python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py`
- Inference - we use Deci's RTiC (RunTime inference container) API for inference of the compiled model.
 
## Quantization
​
We use Intel's OpenVINO Post-Training Optimization (pot) tool.
We run the following command: 
`pot --config CONFIGURATION_PATH.json --evaluate`
We use the DefaultQuantization algorithm with the following configuration:
We calibrate the model on 500 images from the ImageNet validation dataset (given in the MLPerf repo) with a 0.3% accuracy degradation.
​
For further details see [Intel's v0.5 submission](https://github.com/mlperf/inference_results_v0.5/blob/master/closed/Intel/calibration/OpenVino_calibration.md)
and [OpenVino documentation](https://docs.openvinotoolkit.org)