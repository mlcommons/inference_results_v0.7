# where to find stuff
export DATA_ROOT=$HOME/datasets
export MODEL_DIR=$HOME/models
MLPERF_DIR=$HOME/mlperf_inference

# final results go here
export ORG="IVATech"
export DIVISION="closed"
export SUBMISSION_ROOT="/tmp/mlperf-submission"
export SUBMISSION_DIR="$SUBMISSION_ROOT/$DIVISION/$ORG"

# options for official runs
gopt=""


function one_run {
    # args: mode count framework device model ...
    scenario=$1
    count=$2
    model=$3

    system_id="iva-fpga-1"
    echo "====== $model/$scenario ====="

    case $model in 
    mobilenet)
        cmd="$MLPERF_DIR/vision/classification_and_detection/tools/accuracy-imagenet.py --imagenet-val-file $DATA_ROOT/imagenet/val_map.txt"
        offical_name="mobilenet";;
    resnet50) 
        cmd="$MLPERF_DIR/vision/classification_and_detection/tools/accuracy-imagenet.py --imagenet-val-file $DATA_ROOT/imagenet/val_map.txt"
        offical_name="resnet";;
    ssd-mobilenet) 
        cmd="$MLPERF_DIR/vision/classification_and_detection/tools/accuracy-coco.py --coco-dir $DATA_ROOT/coco"
        offical_name="ssd-small";;
    ssd-resnet34) 
        cmd="$MLPERF_DIR/vision/classification_and_detection/tools/accuracy-coco.py --coco-dir $DATA_ROOT/coco"
        offical_name="ssd-large";;
    esac
    output_dir=$SUBMISSION_DIR/results/$system_id/$offical_name

    # performance run
    cnt=0
    while [ $cnt -le $count ]; do
        let cnt=cnt+1
        echo "$scenario performance run $cnt"
        ./run_local.sh -o $output_dir/$scenario/performance/run_$cnt $model $scenario performance $@
    done

    # accuracy run
    ./run_local.sh -o $output_dir/$scenario/accuracy $model $scenario accuracy

    python $cmd --mlperf-accuracy-file $output_dir/$scenario/accuracy/mlperf_log_accuracy.json  \
            --dtype int32  \
            >  $output_dir/$scenario/accuracy/accuracy.txt
    cat $output_dir/$scenario/accuracy/accuracy.txt

    # setup the measurements directory
    mdir=$SUBMISSION_DIR/measurements/$system_id/$offical_name/$scenario
    mkdir -p $mdir
    cp mlperf.conf $mdir

    cat > $mdir/user.conf <<EOF
*.Offline.target_qps = 88.0
*.SingleStream.target_qps = 88.0
*.MultiStream.target_qps = 88.0
*.Server.target_qps = 10.0
EOF
    touch $mdir/README.md
    impid="v1"
    cat > $mdir/$system_id"_"$impid"_"$scenario".json" <<EOF
{
    "input_data_types": "int8",
    "retraining": "N",
    "starting_weights_filename": "https://zenodo.org/record/2535873/files/resnet50_v1.pb",
    "weight_data_types": "int8,fp16",
    "weight_transformations": "quantization"
}
EOF
}

function one_model {
    # args: model ...
    one_run SingleStream 1 $@
    #one_run Server 5 $@
    one_run Offline 1 $@
    #one_run MultiStream 1 $@
}


# run image classifier benchmarks 
export DATA_DIR=$DATA_ROOT/imagenet

one_model resnet50 $gopt

if [ ! -d ${SUBMISSION_DIR}/systems ]; then
    mkdir ${SUBMISSION_DIR}/systems
fi

cp systems/*.json $SUBMISSION_DIR/systems

function compliance() {
  scenario=$1

  #offline
  mkdir -p /tmp/compliance_$scenario && cd /tmp/compliance_$scenario
  mkdir -p TEST01 TEST04-A  TEST04-B  TEST05

  # Run TEST01 compliance check
  cp ~/mlperf_inference/compliance/nvidia/TEST01/resnet50/audit.config TEST01
  cd TEST01
  iva_resnet50 $scenario performance --dataset ~/datasets/imagenet -c ~/mlperf_inference/mlperf.conf -v ~/datasets/imagenet/val_map.txt -m ~/models/resnet50.tpu
  python ~/mlperf_inference/compliance/nvidia/TEST01/run_verification.py  --results_dir /tmp/mlperf-submission/closed/IVATech/results/iva-fpga-1/resnet/$scenario --compliance_dir /tmp/compliance_$scenario/TEST01 --output_dir /tmp/mlperf-submission/closed/IVATech/compliance/iva-fpga-1/resnet/$scenario
  cd -

  for t in TEST04-A  TEST04-B  TEST05; do
    cp ~/mlperf_inference/compliance/nvidia/$t/audit.config $t;
    cd $t;
    iva_resnet50 $scenario performance --dataset ~/datasets/imagenet -c ~/mlperf_inference/mlperf.conf -v ~/datasets/imagenet/val_map.txt -m ~/models/resnet50.tpu
    cd -
  done
  python ~/mlperf_inference/compliance/nvidia/TEST04-A/run_verification.py --test4A_dir TEST04-A --test4B_dir TEST04-B --output_dir /tmp/mlperf-submission/closed/IVATech/compliance/iva-fpga-1/resnet/$scenario
  python ~/mlperf_inference/compliance/nvidia/TEST05/run_verification.py  --results_dir /tmp/mlperf-submission/closed/IVATech/results/iva-fpga-1/resnet/$scenario --compliance_dir /tmp/compliance_$scenario/TEST05 --output_dir /tmp/mlperf-submission/closed/IVATech/compliance/iva-fpga-1/resnet/$scenario
}
#  run compliance

compliance SingleStream
compliance Offline

python ~/mlperf_inference/tools/submission/truncate_accuracy_log.py --input /tmp/mlperf-submission --output /tmp/mlperf-submission-truncated --submitter IVATech
python ~/mlperf_inference/tools/submission/submission-checker.py --input /tmp/mlperf-submission-truncated --version v0.7 --submitter IVATech

