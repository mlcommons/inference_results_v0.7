# MLPerf Inference v0.7 NVIDIA-Optimized Implementations for Open Division

## ResNet50 INT4 Precision

Documentation on the ResNet50 implementation can be found in
[code/resnet50/int4/README.md](code/resnet50/int4/README.md).

This submission is only supported on TitanRTXx4, T4x8, and T4x20 systems.

To run the ResNet50 implementation, first launch the container with:
```
$ make prebuild_int4
```

Then within the container, run:
```
$ make build            # Builds the necessary executables
$ export SYSTEM=<YOUR SYSTEM HERE>      # Needs to be one of: TitanRTXx4, T4x8, T4x20
$ make run_int4_$(SYSTEM)_performance   # Runs the performance run (test_mode=PerformanceOnly)
$ make run_int4_$(SYSTEM)_accuracy      # Runs the accuracy run (test_mode=AccuracyOnly)
```

If this is part of a submission, you will want to run:
```
$ python3 scripts/update_results_int4.py
```
This will export the logs into the `results/` directory. To truncate the accuracy logs, follow the instructions in the
closed submission (`closed/NVIDIA/README.md`).

## BERT-Sparse

Documentation on the Bert-Sparse implementation can be found in
[code/bert/tensorrt_sparse/README.md](code/bert/tensorrt_sparse/README.md).

This submission is only supported on DGX-A100 systems with either 1 GPU or all 8 GPUs enabled.

To run the BERT Sparse implementation, you will want to follow the documentation in its README to download the dataset
and the model, and run the data preprocessing.

Then to launch the container:
```
$ make prebuild_sparse
```

Then within the container, run:
```
$ make build
$ make generate_engines
$ make run_harness RUN_ARGS="--test_mode=PerformanceOnly"
$ make run_harness RUN_ARGS="--test_mode=AccuracyOnly"
```

To update the results, run:
```
$ python3 scripts/update_results.py
```
This will export the logs into the `results/` directory. To truncate the accuracy logs, follow the instructions in the
closed submission (`closed/NVIDIA/README.md`).
