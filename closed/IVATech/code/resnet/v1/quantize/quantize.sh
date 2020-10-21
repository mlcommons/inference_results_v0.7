set -e

if [ ! -f models/resnet50_v1.pb ]; then
	echo "Download Resnet50 model"
	wget -q https://zenodo.org/record/2535873/files/resnet50_v1.pb -O models/resnet50_v1.pb 
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
	$PYTHON make_calibration_set.py --dataset ~/datasets/imagenet --calibration-list models/calibration_list.txt --output models/calibration_tensors_resnet50.npy
fi

if [ ! -f quantized_data/quant_network_graph.pb ]; then
	$QUANTIZE models/resnet50_v1.pb models/calibration_tensors_resnet50.npy --input_nodes=input_tensor --output_nodes=resnet_model/final_dense --save_dir=quantized_data
fi

echo "Quantization pass successful"
