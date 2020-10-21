./Release/ov_mlperf --scenario SingleStream \
	--mode Accuracy \
	--mlperf_conf Configs/mlperf.conf \
	--user_conf Configs/resnet50/user.conf \
	--model_name resnet50 \
	--data_path /home/t/CK-TOOLS/dataset-imagenet-ilsvrc2012-val \
	-nireq 1 \
	--nthreads 28 \
	--nstreams 1 \
	--total_sample_count 50000 \
	--model_path Models/resnet50/resnet50_int8.xml


