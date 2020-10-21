# NVIDIA MLPerf Inference System Under Test (SUT) performance tuning guide

The NVIDIA MLPerf Inference System Under Test implementation has many different parameters which can be tuned to achieve the best performance under the various MLPerf scenarios on a particular system.
However, if starting from a good baseline set of parameters, only a small number of settings will need to be adjusted to achieve good performance.

⚠ **Important**:
Please restrict your performance tuning changes to the settings in the `configs/<BENCHMARK>/<SCENARIO>/config.json` files.
All files in the `measurements/` directory, including `user.conf`, are automatically generated from the `config.json` files.
So please do **not** modify the `user.conf` files.

## Supported systems

We formally support and fully test the configuration files for only the systems listed in [README.md](README.md).
To run on a different system configuration, follow the steps below:

If you plan to run on NVIDIA A100 or NVIDIA T4 with different numbers of GPUs, you can add your own configurations based on the provided ones:

1. Edit `configs/<BENCHMARK>/<SCENARIO>/config.json` and add a configuration for your system.
To do this, you can copy an existing configuration for a similar system and update the relevant fields.
2. Modify the `offline_expected_qps`, `server_target_qps`, and `multi_stream_samples_per_query` fields by scaling them with the number of GPUs you use.
For example, if you plan to run on a T4x4 system and use the T4x8 configuration as baseline, then scale these values by 0.5.
3. For BERT benchmark, `server_num_issue_query_threads` field also need to be scaled by the number of GPUs.
4. Add a new line in [code/common/system_list.py](code/common/system_list.py) with your GPU name and the number of GPUs.
5. Run commands as documented in [README.md](README.md).
It should work or should be very close to working.
6. If you see INVALID results in any case, follow the steps in the [Fix INVALID results](#fix-invalid-results) section below.

If you plan to run on GPUs other than NVIDIA A100 or NVIDIA T4, you can still follow these steps to get a set of baseline config files.
However, more performance tuning will be required to achieve better performance numbers, such as changing batch sizes or number of streams to run inference with.
See the [Tune parameters for better performance](#tune-parameters-for-better-performance) section below.

## Test runtime

The [MLPerf Inference rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc) requires each submission to meet certain requirements to become a valid submission, and one of them is the runtime for the benchmark test.
Below we summarize the expected runtime for each scenario.
Some of them will take several hours to run, so we also provide tips to reduce test runtime for faster debugging cycles.

### Offline

Default test runtime: `max((1.1 * min_duration * offline_expected_qps / actual_qps), (min_query_count / actual_qps))`, where:

- `min_duration`: 60 seconds by default.
- `offline_expected_qps`: set by `config.json`.
- `min_query_count`: 24576 by default.

The typical runtime for Offline scenario should be about 66 seconds, unless on a system with very low QPS, in which case, runtime will be much longer.
To reduce runtime for debugging, simply lower `min_duration` and `min_query_count` by adding flags like `--min_duration=1000 --min_query_count=1` to `RUN_ARGS` in the command.

### Server

Default test runtime: `max((min_duration * server_target_qps / actual_qps), (min_query_count / actual_qps))`, where:

- `min_duration`: 60 seconds by default.
- `server_target_qps`: set by `config.json`.
- `min_query_count`: 270336 by default.

The typical runtime for Server scenario is about 60 seconds if `server_target_qps` is equal to or is lower than actual QPS.
Otherwise, the runtime will be longer since queries start to queue up as the system cannot digest the queries in time.
To reduce runtime for debugging, simply lower `min_duration` and `min_query_count` by adding flags like `--min_duration=1000 --min_query_count=1` to `RUN_ARGS` in the command.

### SingleStream

Default test runtime: `max((min_duration / single_stream_expected_latency_ns * actual_latency), (min_query_count * actual_latency))`, where:

- `min_duration`: 60 seconds by default.
- `single_stream_expected_latency_ns`: set by `config.json`.
- `min_query_count`: 1024 by default.

The typical runtime for SingleStream scenario should be about 60 seconds, unless on a system with very long latency per sample, in which case, runtime will be much longer.
To reduce runtime for debugging, simply lower `min_duration` and `min_query_count` by adding flags like `--min_duration=1000 --min_query_count=1` to `RUN_ARGS` in the command.

### MultiStream

Default test runtime: `max((min_duration), (min_duration / multi_stream_target_qps * multi_stream_samples_per_query / actual_qps), (min_query_count / multi_stream_target_qps))`, where:

- `min_duration`: 60 seconds by default.
- `multi_stream_target_qps`: 20 qps for ResNet50 and SSDMobileNet benchmarks and 16 qps for SSDResNet34 benchmark. Do not change this since this is defined by MLPerf Inference rules.
- `multi_stream_samples_per_query`: set by `config.json`.
- `min_query_count`: 270336 by default.

Note that this results in typical test runtime being roughly 3h45m for ResNet50 and SSDMobileNet benchmarks and 4h45m for SSDResNet34 benchmark regardless of the actual QPS of the system.
To reduce runtime for debugging, simply lower `min_duration` and `min_query_count` by adding flags like `--min_duration=1000 --min_query_count=1` to `RUN_ARGS` in the command.

## Fix INVALID results

### Offline

The most common reason for INVALID results in Offline scenario is that the actual QPS of the system is much higher than the `offline_expected_qps`.
Therefore, simply increase `offline_expected_qps` until the max query latency reported by LoadGen reaches 66 secs, which is when `offline_expected_qps` matches the actual QPS.

### Server

The most common reason for INVALID results in Server scenario is that the actual QPS of the system is lower than the `server_target_qps`.
Therefore, reduce `server_target_qps` until the 99th percentile latency falls below the Server latency targets defined by the [MLPerf Inference rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc). If lowering `server_target_qps` does not reduce the 99th percentile latency,
try reducing `gpu_batch_size` and/or `gpu_inference_streams` instead.

### SingleStream

The most common reason for INVALID results in SingleStream scenario is that the actual latency of the system for each sample is much lower than the `single_stream_expected_latency_ns`.
Therefore, simply lower `single_stream_expected_latency_ns` to match the actual 90th percentile latency reported by LoadGen.

### MultiStream

The most common reason for INVALID results in MultiStream scenario is that the actual QPS of the system cannot handle the target `multi_stream_samples_per_query` we set.
Therefore, reduce `multi_stream_samples_per_query` until the 99th percentile falls below the MultiStream target QPS defined by [MLPerf Inference rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc).

## Tune parameters for better performance

To get better performance numbers, parameters in the config files such as batch sizes can be further tuned.
All settings can be adjusted using the configuration files. Additionally, for interactive experiments, the settings can be adjusted on the command line.

For example:

```
$ make run RUN_ARGS="--benchmarks=ResNet50 --scenarios=offline --gpu_batch_size=128"
```

will run ResNet50 in the offline scenario, overriding the batch size to 128.
To use that batch size for submission, the `gpu_batch_size` setting will need to be changed in the corresponding config.json file.

Below we show some common parameters that can be tuned if necessary.
There are also some benchmark-specific or platform-specific parameters which are not listed here.
For example, on Jetson AGX Xavier or Xavier NX platforms for ResNet50/SSDResNet34/SSDMobileNet, there are also DLA parameters (such as `dla_batch_size`) that can be tuned for performance.
Please look at the code to understand the purpose of those parameters.

### Offline

In the Offline scenario, the default settings should provide good performance.
Optionally, try increasing `gpu_batch_size` to see if it gives better performance. CUDA stream settings like `gpu_copy_streams` and `gpu_inference_streams` can also be tried to better overlap the memory transfers with inference computation.

### Server

Server scenario tuning can be a bit more involved.
The goal is to increase `server_target_qps` to the maximum value that can still satisfy the latency requirements specified by the [MLPerf Inference rules](https://github.com/mlperf/inference_policies/blob/master/inference_rules.adoc).

This is typically an iterative process:

1. Increase `server_target_qps` to the maximum value that still meets the latency constraints with the current settings
2. Sweep across many variations of the existing settings, such as `gpu_batch_size`, `deque_timeout_us`, `gpu_copy_streams`, `gpu_inference_streams`, and so on.
3. Replace the current settings with those providing the lowest latency at the target percentile, with the current `server_target_qps` setting
4. Goto 1.

### SingleStream

In the SingleStream scenario, the default settings should provide good performance.

### MultiStream

The multistream scenario is conceptually simpler than the server scenario, but the tuning process is similar.
The goal is to increase the `gpu_multi_stream_samples_per_query` to the largest value that still satisfies the latency constraints.

To start, set `gpu_batch_size` to be equal to `gpu_multi_stream_samples_per_query` and then increase the values to the maximum point where the latency constraint is still met.

Since all samples belonging to a query arrive at the same time, use of a single batch to process the entire query will lead to serialization between the data copies between the host and device and the inference.
This can be mitigated by splitting the query into multiple batches, reducing the amount of data that must be transferred before inference can begin on the device.
If `gpu_batch_size` is less than `gpu_multi_stream_samples_per_query`, the samples will be divided into `ceil(gpu_multi_stream_samples_per_query / gpu_batch_size)` batches.
The data transfers can then be pipelined against computation, reducing overall latency. Other settings like `gpu_copy_streams` and `gpu_inference_streams` can also be tried.

## Other performance tips

- For systems with passive cooling GPUs, especially systems with NVIDIA T4, the cooling system plays an important role in performance.
You can run `nvidia-smi dmon -s pc` to monitor the GPU temperature while harness is running.
To get best performance, GPU temperature should saturate to a reasonable temperature, such as 65C, instead of constantly going up and throttling the GPU clock frequencies.
On Jetson platforms, use `tegrastats` to monitor GPU temperature instead.
- In Server scenario, please make sure that the Transparent Huge Page setting is set to "always".
