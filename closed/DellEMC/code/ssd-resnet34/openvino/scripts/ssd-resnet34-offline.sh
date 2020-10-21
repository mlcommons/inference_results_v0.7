./Release/ov_mlperf --scenario Offline \
	--mlperf_conf Configs/mlperf.conf \
	--user_conf Configs/ssd-resnet34/user.conf \
	--model_name ssd-resnet34 \
	--data_path /home/dell/CK-TOOLS/dataset-coco-2017-val \
	--nireq 56 \
	--nthreads 56 \
	--nstreams 56 \
	--model_path Models/ssd-resnet34/ssd-resnet34_int8.xml

