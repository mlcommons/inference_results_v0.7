
if [ -d build ]; then
	rm -r build
fi

mkdir build && cd build

source /opt/intel/openvino/bin/setupvars.sh

cmake -DInferenceEngine_DIR=/opt/intel/openvino/inference_engine/share \
		-DLOADGEN_DIR=/home/t/MLPerf/mlperf-tracking/inference/loadgen \
		-DBOOST_INCLUDE_DIRS=/home/t/Downloads/boost_1_72_0 \
		-DBOOST_FILESYSTEM_LIB=/home/t/Downloads/boost_1_72_0/stage/lib/libboost_filesystem.so \
		-DIE_SRC_DIR=/opt/intel/openvino/inference_engine/src \
		-DCMAKE_BUILD_TYPE=Release \
		..

make
