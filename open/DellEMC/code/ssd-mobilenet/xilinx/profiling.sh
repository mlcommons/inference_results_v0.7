export MLPERF_RUNTIME=1
export MLPERF_POST=1
export MLPERF_SORT=0

./run_singlestream.sh model_ssd_mobilenet 2&>/dev/shm/log_mlperf
./average.sh
#export MLPERF_RUNTIME=0
#export MLPERF_POST=0
#export MLPERF_SORT=1
#./run_singlestream.sh model_ssd_mobilenet 2&>/dev/shm/log_mlperf_sort
#./average_sort.sh

