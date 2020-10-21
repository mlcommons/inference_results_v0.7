# MLPerf Inference v0.7 Submission Guide

The general steps for submission are the same as documented in [the closed
submission](../../closed/NVIDIA/submission_guide.md). However, there are a few things you should note:

 1. Running the ResNet50 INT4 harness will cause the working directory to change to `code/resnet50/int4`. As such,
    `audit.config` must be placed into `code/resnet50/int4` for audit tests to run correctly.
 2. `make truncate_results` must be run from `closed/NVIDIA`, not `open/NVIDIA`.
 3. There are separate scripts for `update_results` for ResNet50 INT4 and BERT Sparse: Use `python3
    scripts/update_results_int4.py` for ResNet50 INT4, and `python3 scripts/update_results.py` for BERT Sparse.
 4. Compliance / Audit tests are not yet automated in the open submission. See the instructions below for instructions
    on running the compliance tests.

## Running audit tests for ResNet50 INT4

 1. Launch the container with `make prebuild_int4`
 2. `make build` to build the executables. Make sure you ran `git lfs pull` if you downloaded the repo via `git clone`
    so `int4_offline.a` is valid.
 3. Run the accuracy and performance runs: `make run_int4_${SYSTEM_ID}_${TEST_MODE}` where `${SYSTEM_ID}` is `T4x8`,
    `T4x20`, or `TitanRTXx4`, and `${TEST_MODE}` is `accuracy` or `performance`.
 4. Run `python3 scripts/update_results_int4.py` to update `results/`.
 5. Run the audit tests:

```
# Run TEST01
cp build/inference/compliance/nvidia/TEST01/resnet50/audit.config code/resnet50/int4/audit.config
make run_int4_${SYSTEM}_performance LOG_DIR=/work/build/TEST01
python3 build/inference/compliance/nvidia/TEST01/run_verification.py --results=results/${SYSTEM}/resnet50/Offline/ --compliance=build/TEST01/${SYSTEM}/resnet/Offline/performance/run_1/ --output_dir=compliance/${SYSTEM}/resnet50/Offline/
# Cleanup
rm -f verify_accuracy.txt verify_performance.txt code/resnet50/int4/audit.config

# Run TEST04
cp build/inference/compliance/nvidia/TEST04-A/audit.config code/resnet50/int4/audit.config
make run_int4_$(SYSTEM)_performance LOG_DIR=/work/build/TEST04-A
cp build/inference/compliance/nvidia/TEST04-B/audit.config code/resnet50/int4/audit.config
make run_int4_$(SYSTEM)_performance LOG_DIR=/work/build/TEST04-B
python3 build/inference/compliance/nvidia/TEST04-A/run_verification.py --test4A_dir build/TEST04-A/${SYSTEM}/resnet/Offline/performance/run_1/ --test4B_dir build/TEST04-B/${SYSTEM}/resnet/Offline/performance/run_1/ --output_dir=compliance/${SYSTEM}/resnet50/Offline/
rm -f verify_accuracy.txt verify_performance.txt code/resnet50/int4/audit.config

# Run TEST05
cp build/inference/compliance/nvidia/TEST05/audit.config code/resnet50/int4/audit.config
make run_int4_$(SYSTEM)_performance LOG_DIR=/work/build/TEST05
python3 build/inference/compliance/nvidia/TEST05/run_verification.py --results_dir=results/${SYSTEM}/resnet50/Offline/ --compliance_dir=build/TEST05/${SYSTEM}/resnet/Offline/performance/run_1/ --output_dir=compliance/${SYSTEM}/resnet50/Offline/
```

 6. Run `make truncate_results` from `closed/NVIDIA`. This will back up the full accuracy logs to
    `closed/NVIDIA/build/full_results`. If that command fails with a file not found, you might need to run `make
    clone_loadgen` from `closed/NVIDIA`.

## Running audit tests for BERT Sparse

 1. Launch the container with `make prebuild_sparse`
 2. `make build`. If this fails during a linking phase due to `Permission Denied`, make sure the permissions in
    `code/bert/tensorrt_sparse/TensorRT-preview/lib` for the `*.so*` files are `0755`, and relaunch the container.
 3. Build the engine with `make generate_engines`. This will take somewhere in the ballpark of 30 min.
 4. Run the accuracy and performance runs: `make run_harness`, `make run_harness RUN_ARGS="--test_mode=AccuracyOnly"`.
 5. Run `python3 scripts/update_results.py` to update `results/`.
 6. Run the audit tests, where `${SYSTEM}` is `A100x1` or `A100x8`, and `<SYSTEM NAME>` is the full name
    (`DGX-A100_A100-SXM4...`):

```
# Run TEST01
cp build/inference/compliance/nvidia/TEST01/bert/audit.config .
make run_harness LOG_DIR=/work/build/${SYSTEM}-audit/TEST01 # This should fail the accuracy test!
# Run TEST01 accuracy fallback
# Generate baseline json
bash build/inference/compliance/nvidia/TEST01/create_accuracy_baseline.sh path/to/original/accuracy/run/mlperf_log_accuracy.json build/${SYSTEM}-audit/TEST01/<SYSTEM ID>/bert-99/Offline/mlperf_log_accuracy.json
# Generate baseline_accuracy.txt
python3 build/inference/language/bert/accuracy-squad.py \
    --log_file mlperf_log_accuracy_baseline.json \
    --vocab_file build/models/bert/vocab.txt \
    --val_data build/data/squad/dev-v1.1.json \
    --out_file build/${SYSTEM}-audit/TEST01/<SYSTEM ID>/bert-99/Offline/predictions.json \
    --output_dtype float16 > baseline_accuracy.txt
# Generate compliance_accuracy.txt
python3 build/inference/language/bert/accuracy-squad.py \
    --log_file build/${SYSTEM}-audit/TEST01/<SYSTEM ID>/bert-99/Offline/mlperf_log_accuracy.json \
    --vocab_file build/models/bert/vocab.txt \
    --val_data build/data/squad/dev-v1.1.json \
    --out_file build/${SYSTEM}-audit/TEST01/<SYSTEM ID>/bert-99/Offline/predictions.json \
    --output_dtype float16 > build/${SYSTEM}-audit/TEST01/<SYSTEM ID>/bert-99/Offline/compliance_accuracy.txt
# Generate verify_accuracy.txt. This command will crash, but verify_accuracy.txt will be in the working directory
python3 build/inference/compliance/nvidia/TEST01/run_verification.py --results=results/<SYSTEM ID>/bert-99/Offline/ \
    --compliance=build/${SYSTEM}-audit/TEST01/<SYSTEM ID>/bert-99/Offline/ \
    --output_dir=compliance/<SYSTEM ID>/bert-99/Offline/
# Generate verify_performance.txt in working directory
python3 build/inference/compliance/nvidia/TEST01/verify_performance.py \
    -r results/<SYSTEM ID>/bert-99/Offline/performance/run_1/mlperf_log_summary.txt \
    -t build/${SYSTEM}-audit/TEST01/<SYSTEM ID>/bert-99/Offline/mlperf_log_summary.txt
```

You will need to manually populate compliance/.../TEST01, based on the directory structure found in [submission rules](https://github.com/mlperf/policies/blob/master/submission_rules.adoc#562-inference).

```
# Run TEST05 (TEST04 is not applicable to BERT)
cp build/inference/compliance/nvidia/TEST05/audit.config .
make run_harness RUN_ARGS="--test_mode=PerformanceOnly" LOG_DIR=/work/build/${SYSTEM}-audit/TEST05
python3 build/inference/compliance/nvidia/TEST05/run_verification.py \
    --results_dir=results/<SYSTEM ID>/bert-99/Offline/ \
    --compliance_dir=build/${SYSTEM}-audit/TEST05/<SYSTEM ID>/bert-99/Offline/ \
    --output_dir=compliance/<SYSTEM ID>/bert-99/Offline/
```

 7. Run `make truncate_results` from `closed/NVIDIA`. This will back up the full accuracy logs to
    `closed/NVIDIA/build/full_results`. If that command fails with a file not found, you might need to run `make
    clone_loadgen` from `closed/NVIDIA`.
