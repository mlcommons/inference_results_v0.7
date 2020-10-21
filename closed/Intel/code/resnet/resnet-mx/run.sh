#!/bin/bash
export OMP_NUM_THREADS=$CPUS_PER_INSTANCE
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

if [ "x$BATCH_SIZE" == "x" ]; then
    echo "BATCH_SIZE not set" && exit 1
fi
if [ "x$NUM_INSTANCE" == "x" ]; then
    echo "NUM_INSTANCE not set" && exit 1
fi
if [ "x$MODEL_PREFIX" == "x" ]; then
    echo "MODEL_PATH not set" && exit 1
fi
if [ "x$DATASET_PATH" == "x" ]; then
    echo "DATASET_PATH not set" && exit 1
fi
if [ "x$DATASET_LIST" == "x" ]; then
    echo "DATASET_LIST not set" && exit 1
fi

NUM_PHY_CPUS=$(( $NUM_INSTANCE*$CPUS_PER_INSTANCE ))

SYMBOL_FILE=$MODEL_PREFIX-symbol.json
PARAM_FILE=$MODEL_PREFIX-0000.params
if [ $# == 1 ]; then
    if [ $1 == "offline" ]; then
        echo "Running offline performance mode"
        python  python/main.py --symbol-file=$SYMBOL_FILE --param-file=$PARAM_FILE --batch-size=${BATCH_SIZE} \
            --num-instance=$NUM_INSTANCE --num-phy-cpus=$NUM_PHY_CPUS --dataset-path=$DATASET_PATH --dataset-list=$DATASET_LIST \
            --cache=1 --use-int8-dataset \
            --mlperf-conf=mlperf.conf --user-conf=user_perf.conf --scenario=Offline
    elif [ $1 == "server" ]; then
        echo "Running sever performance mode"
        python  python/main.py --symbol-file=$SYMBOL_FILE --param-file=$PARAM_FILE --batch-size=${BATCH_SIZE} \
            --num-instance=$NUM_INSTANCE --num-phy-cpus=$NUM_PHY_CPUS --dataset-path=$DATASET_PATH --dataset-list=$DATASET_LIST \
            --cache=1 --use-int8-dataset \
            --mlperf-conf=mlperf.conf --user-conf=user_perf.conf --scenario=Server
    else
        echo "Only offline/server are valid"
    fi
elif [ $# == 2 ]; then
    if [ $1 == "offline" ] && [ $2 == "accuracy" ]; then
        echo "Running offline accuracy mode"
        python  python/main.py --symbol-file=$SYMBOL_FILE --param-file=$PARAM_FILE --batch-size=${BATCH_SIZE} \
            --num-instance=$NUM_INSTANCE --num-phy-cpus=$NUM_PHY_CPUS --dataset-path=$DATASET_PATH --dataset-list=$DATASET_LIST \
            --cache=1 --use-int8-dataset \
            --mlperf-conf=mlperf.conf --user-conf=user_accu.conf --scenario=Offline --accuracy
    elif [ $1 == "server" ] && [ $2 == "accuracy" ]; then
        echo "Running sever accuracy mode"
        python  python/main.py --symbol-file=$SYMBOL_FILE --param-file=$PARAM_FILE --batch-size=${BATCH_SIZE} \
            --num-instance=$NUM_INSTANCE --num-phy-cpus=$NUM_PHY_CPUS --dataset-path=$DATASET_PATH --dataset-list=$DATASET_LIST \
            --cache=1 --use-int8-dataset \
            --mlperf-conf=mlperf.conf --user-conf=user_accu.conf --scenario=Server --accuracy
    else
        echo "Only offline/server accuray are valid"
    fi
else
    echo "Only 1/2 parameters are valid"

fi
