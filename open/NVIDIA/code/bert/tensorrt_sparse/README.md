# Sparse BERT Benchmark Setup and Usage

This benchmark performs language processing using a BERT network.

## Environment setup

The steps to create the environment and build the harness are described in the [General Instructions](../../../../../closed/NVIDIA/README.md)
and [MLPerf Inference v0.7 NVIDIA-Optimized Implementations for Open Division](../../../README.md).

## Model

This model differs from  `bert_large_v1_1.onnx` in Zenodo and linked in the MLPerf inference repository as follows:
* Residual paths start before layer-norm, a model architecture that’s been shown to be more stable for large models [1,2].
* Model parameter tensors (weights in QKV, Proj, FF1, and FF2 layers) use 2:4 structured sparsity [3].

### Generating the Sparse INT8 Model

This submission uses Megatron [4] to pre-train a BERT-Large model (continuing
to fine-tune this model for the SQuAD data set results in an F1 score of 91.1).
Then, the ASP library [5] is used to prune 50% of the weights using the 2:4 sparsity
pattern (at least 2 weights out of each group of 4 must be zero), and
pre-train the model once more with the same hyper-parameters as used for the
dense pre-training phase. The sparse pre-trained model is then simultaneously
quantized and fine-tuned on SQuAD to a final F1 score of 91.3 following the
techniques in [6] with the following choices and hyperparameters:
* Residual connections are quantized.
* QA heads (the last layer used for classification in SQuAD) are not quantized.
* SQuAD fine-tuning is done in a single pass, with quantization.
* Activations use per-tensor 99.99% percentile calibration.
* Weights use per-tensor max calibration.
* Fine-tuning is done over 2 epochs on 8 GPUs with batch size 2 per GPU; the learning rate
follows a linear warmup to 4e-5 for the first 10% of iterations then follows a
monotonically decreasing cosine-annealing schedule down to 0.

### Downloading / obtaining the model

The model `megatron.pkl` is downloaded by running `bash code/bert/tensorrt/download_model.sh`.
The model is imported into a TensorRT network by reading the serialized weights and parameters of the model,
construcing the network description, and finally building the corresponding TensorRT plan.
Details can be found in [bert_var_seqlen.py](bert_var_seqlen.py).

## Optimizations

In addition to the optimizations described in the [BERT benchmark Setup and Usage](../../closed/NVIDIA/code/bert/tensorrt/README.md), this benchmark
is optimized as described in the following subsections.

### Plugins

The following TensorRT plugins are used to optimize BERT benchmark:
- `CustomEmbLayerNormPluginDynamic` version 3: optimizes fused embedding table lookup and LayerNorm operations.
- `CustomSkipLayerNormPluginDynamic` version 3 and 4: optimizes fused LayerNorm operation and residual connections.
- `CustomQKVToContextPluginDynamic` version 3: optimizes fused Multi-Head Attentions operation.

These plugins are a preview of the version that will be available in a future TensorRT release.

### Sparsity

As mentioned above, this model is trained to have 2:4 structured sparsity and leverage sparsity support.

### References
1. [Language Models are Unsupervised Multitask Learners, Radford et al.](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
2. [Learning Deep Transformer Models for Machine Translation, Wang et al.](https://www.aclweb.org/anthology/P19-1176/)
3. [Accelerating Sparsity in the NVIDIA Ampere Architecture, NVIDIA](https://resources.nvidia.com/events/GTC2020s22085)
4. [Megatron-LM, NVIDIA](https://github.com/NVIDIA/Megatron-LM)
5. [Automatic Sparsity, NVIDIA](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity)
6. [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation, Wu et al.](https://arxiv.org/abs/2004.09602)
