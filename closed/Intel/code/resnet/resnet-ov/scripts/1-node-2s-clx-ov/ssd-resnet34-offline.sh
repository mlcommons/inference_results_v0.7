./Release/ov_mlperf --scenario Offline \
	--mlperf_conf Configs/mlperf.conf \
	--user_conf Configs/ssd-resnet34/user.conf \
	--model_name ssd-resnet34 \
	--data_path /home/t/CK-TOOLS/dataset-coco-2017-val \
	--nireq 56 \
	--nthreads 56 \
	--nstreams 56 \
	--total_sample_count 5000 \
	--warmup_iters 500 \
	--model_path Models/ssd-resnet34/ssd-resnet34_int8.xml

