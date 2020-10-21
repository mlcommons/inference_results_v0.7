# Xilinx MLperf Vitis submission

Systems:
* R740xd_u280_v3me
  - SSD Mobilenet v1
  - SingleStream scenario
  - Open division
  - features: DPU (for SSD MobileNet acceleration) + HLS kernel (for post-processing acceleration), ~1.3ms end-to-end latency for batch 1
* R740xd_vck5000 
  - Resnet50 (pruned)
  - Server scenario
  - Offline scenario
  - Open division (rdi)
  - features: 350MHz DPU using 1.33GHz Xilinx AI Engine, <2ms DPU latency for batch 8

Reference: 
* https://github.com/mlperf/policies/blob/master/submission_rules.adoc#562-inference

Submission scripts:
* `python3 mlperf-inference/tools/submission/truncate_accuracy_log.py --input . --submitter Xilinx --output truncated` (to get accuracy.txt with hash)
* `python3 mlperf-inference/tools/submission/submission-checker.py --input .`
