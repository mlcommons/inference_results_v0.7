./Release/ov_mlperf --scenario Offline \
	--mlperf_conf Configs/mlperf.conf \
	--user_conf Configs/resnet50/user.conf \
	--model_name resnet50 \
	--data_path /home/dell/CK-TOOLS/dataset-imagenet-ilsvrc2012-val \
	--nireq 56 \
	--nthreads 56 \
	--nstreams 56 \
  --model_path Models/resnet50/resnet50_int8.xml



