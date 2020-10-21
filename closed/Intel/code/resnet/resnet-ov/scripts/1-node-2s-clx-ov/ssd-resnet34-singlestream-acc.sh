./Release/ov_mlperf --scenario SingleStream \
        --mode Accuracy \
	--mlperf_conf Configs/mlperf.conf \
	--user_conf Configs/ssd-resnet34/user.conf \
	--model_name ssd-resnet34 \
	--data_path /home/t/CK-TOOLS/dataset-coco-2017-val \
	--nireq 1 \
	--nthreads 28 \
	--nstreams 1 \
	--total_sample_count 5000 \
	--warmup_iters 500 \
	--model_path Models/ssd-resnet34/ssd-resnet34_int8.xml

