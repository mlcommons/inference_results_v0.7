for i in $(seq 1 5)
do
mkdir result/single/$i
./run_singlestream.sh model_ssd_mobilenet 
cat mlperf_log_summary.txt | grep "90th percentile latency"
mv mlperf_log* result/single/$i
done
