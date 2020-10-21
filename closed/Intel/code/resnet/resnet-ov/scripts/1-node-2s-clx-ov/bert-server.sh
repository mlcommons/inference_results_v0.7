echo "Test start time: $(date)"
Release/ov_mlperf --scenario Server --mlperf_conf Configs/mlperf.conf --user_conf Configs/bert/user.conf --model_name bert --data_path /root/cbai/squadv1.1 --nseq 128 --nseq_step 64 --nireq 4 --nthreads 56 --nstreams 4 --total_sample_count 10833 --warmup_iters 100 --model_path Models/bert/bert_large_v1_1_int8.xml
echo "Test end time: $(date)"
