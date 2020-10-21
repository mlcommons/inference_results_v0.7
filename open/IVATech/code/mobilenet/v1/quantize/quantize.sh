set -e

if [ ! -f models/mobilenet_v1_1.0_224_frozen.pb ]; then
	echo "Download MobileNet V1 model"
	wget -q https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224.tgz -O models/mobilenet_v1_1.0_224.tgz
	tar -C models -zxf models/mobilenet_v1_1.0_224.tgz ./mobilenet_v1_1.0_224_frozen.pb
fi

if [ ! -d .venv ]; then
       echo "Create Python3 Environment"	
       python3 -mvenv .venv
       ./.venv/bin/pip install -U pip setuptools
       ./.venv/bin/pip install -r requirements.txt
fi

PYTHON=./.venv/bin/python
PIP=./.venv/bin/pip
QUANTIZE=./.venv/bin/quantize_graph

if [ ! -f models/calibration_tensors_resnet50.npy ]; then
	wget https://raw.githubusercontent.com/mlperf/inference/master/calibration/ImageNet/cal_image_list_option_1.txt -O models/calibration_list.txt
	$PYTHON make_calibration_set.py --dataset ~/datasets/imagenet --calibration-list models/calibration_list.txt --output models/calibration_tensors_mobilenet_v1.npy
fi

if [ ! -f quantized_data/quant_network_graph.pb ]; then
	$QUANTIZE models/mobilenet_v1_1.0_224_frozen.pb models/calibration_tensors_mobilenet_v1.npy --float16_capacity_control=true --input_nodes=input --output_nodes=MobilenetV1/Logits/SpatialSqueeze --save_dir=quantized_data
fi

echo "Quantization pass successful"
