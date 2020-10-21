for i in $(seq 1 20)
do
mkdir result/server/$i
./run_server.sh 285
cat mlperf_log_summary.txt | grep "Result is"
cat mlperf_log_summary.txt | grep "Scheduled samples per second"
cat mlperf_log_summary.txt | grep "99.00 percentile latency"
mv mlperf_log_* result/server/$i
done
