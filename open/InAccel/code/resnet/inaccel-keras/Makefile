.ONESHELL: loadgen classification_and_detection

all: get setup patch

get:
	git submodule update --init --recursive

setup: loadgen classification_and_detection

loadgen:
	cd inference/loadgen
	CFLAGS="-std=c++14" python setup.py develop --user

classification_and_detection:
	cd inference/vision/classification_and_detection
	python setup.py develop --user

patch:
	patch --strip 0 < mlperf.patch
