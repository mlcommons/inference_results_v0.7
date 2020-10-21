#!/bin/bash

if [ -z "$MLPERF_INFERENCE" ]
then
COMPLIANCE_DIR="../mlperf-inference/compliance/nvidia"
else
COMPLIANCE_DIR="${MLPERF_INFERENCE}/compliance/nvidia"
fi

RUN_PARAMS="" # params passed to ./run.sh

while true
do
  if [[ -z "$1" ]]; then break; fi
  case "$1" in
    -p  |--params             ) RUN_PARAMS="$2"        ; shift 2 ;;
     *) echo "Unknown argument : $1";
        exit 1 ;;
  esac
done
for SCENARIO in SingleStream Offline
do
  source /opt/xilinx/xrt/setup.sh
  RESULT_DIR="results/ssd-small/${SCENARIO}"
  AUDIT_DIR="audit/ssd-small/${SCENARIO}"
  if [ ${SCENARIO} = "Offline" ]
  then
    #cp model_ssd_mobilenet/meta.json.multi model_ssd_mobilenet/meta.json
    NUM_ITER=1
    RUN_CMD=offline
  else
    cp model_ssd_mobilenet/meta.json.single model_ssd_mobilenet/meta.json
    NUM_ITER=1
    RUN_CMD=singlestream
  fi
  for (( ITER=1; ITER<=$NUM_ITER; ITER++ ))
  do
    mkdir -p ${RESULT_DIR}/performance/run_${ITER}
    echo -e "\nRunning ${SCENARIO} performance #${ITER}"
    ./run_${RUN_CMD}.sh model_ssd_mobilenet ${RUN_PARAM}
    cp mlperf_log*.{txt,json} ${RESULT_DIR}/performance/run_${ITER}
  done
  mkdir -p ${RESULT_DIR}/accuracy
  echo -e "\nRunning ${SCENARIO} accuracy"
  ./run_${RUN_CMD}_accuracy.sh model_ssd_mobilenet ${RUN_PARAM}
  cp mlperf_log*.{txt,json} accuracy.txt ${RESULT_DIR}/accuracy


  # TEST01
  echo -e "\nRunning ${SCENARIO} TEST01 compliance"
  mkdir -p ${AUDIT_DIR}/TEST01/{performance,accuracy}
  cp ${COMPLIANCE_DIR}/TEST01/ssd-mobilenet/audit.config .
  ./run_${RUN_CMD}.sh model_ssd_mobilenet ${RUN_PARAM}
  cp mlperf_log*.{txt,json} accuracy.txt ${AUDIT_DIR}/TEST01/accuracy
  cp mlperf_log*.{txt,json} ${AUDIT_DIR}/TEST01/performance
  python3 ${COMPLIANCE_DIR}/TEST01/run_verification.py -r ${RESULT_DIR} -o ${AUDIT_DIR} --dtype float32 -c ${AUDIT_DIR}/TEST01/performance
  rm audit.config
  
  #TEST04-A and TEST04-B
  for item in TEST04-A TEST04-B 
  do
    mkdir -p ${AUDIT_DIR}/${item}/{performance,accuracy}
    echo -e "\nRunning ${SCENARIO} ${item}"
    cp ${COMPLIANCE_DIR}/${item}/audit.config .
    ./run_${RUN_CMD}.sh model_ssd_mobilenet ${RUN_PARAM} 
    cp mlperf_log*.{txt,json} ${AUDIT_DIR}/${item}/performance
    rm audit.config
  done
  echo "Running TEST04-A TEST04-B compliance"
  python3 ${COMPLIANCE_DIR}/TEST04-A/run_verification.py -a ${AUDIT_DIR}/TEST04-A/performance -b ${AUDIT_DIR}/TEST04-B/performance -o ${AUDIT_DIR} 

  # TEST05 
  echo -e "\nRunning ${SCENARIO} TEST05 compilance"
  mkdir -p ${AUDIT_DIR}/TEST05/{performance,accuracy}
  cp ${COMPLIANCE_DIR}/TEST05/audit.config .
  ./run_${RUN_CMD}.sh model_ssd_mobilenet ${RUN_PARAM}
  cp mlperf_log*.{txt,json} ${AUDIT_DIR}/TEST05/accuracy
  cp mlperf_log*.{txt,json} ${AUDIT_DIR}/TEST05/performance
  python3 ${COMPLIANCE_DIR}/TEST05/run_verification.py -r ${RESULT_DIR} -o ${AUDIT_DIR} -c ${AUDIT_DIR}/TEST05/performance
  rm audit.config
done
