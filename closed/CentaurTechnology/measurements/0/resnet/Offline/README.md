Begin from the `/workspace` directory within the docker environment provided (see `closed/CentaurTechnology/measurements/README.md` for details).

Install the Ncore TF-Lite interface python module `closed/CentaurTechnology/code/tflite-code/ncoretflite-0.1-cp36-cp36m-linux_x86_64.whl`.

```
python3.6 -m pip install closed/CentaurTechnology/code/tflite-code/ncoretflite-0.1-cp36-cp36m-linux_x86_64.whl
```

Download the Ncore TF delegate library `ncore_tf_delegate.so` from `https://www.dropbox.com/s/5sj2qguyvonpl4c/ncore_tf_delegate.so?dl=0` (library is too large to push to the MLPerf submission repo), and place it in the code directory.

Download the quantized model `resnet50_v1.tflite` from `https://www.dropbox.com/s/3rgtr3vc99ufttp/resnet50_v1.tflite?dl=0`.

Use the following commands for performance or accuracy runs.

## Performance
```
python python/main.py \
    --backend=tflite-ncore-resnet-offline \
    --cache=1 \
    --mlperf_conf=../../mlperf.conf \
    --user_conf=user.conf \
    --dataset-path=/datasets/mlperf-v0.5/dataset-imagenet-ilsvrc2012-val \
    --max-batchsize=1024 \
    --model=resnet50_v1.tflite \
    --model-name=resnet50 \
    --profile=resnet50-tf-ncore-offline \
    --scenario=Offline \
    --threads=2 \
    --output=/tmp/mlperf-tempout/output
```

## Accuracy
```
python python/main.py \
    --backend=tflite-ncore-resnet-offline \
    --cache=1 \
    --mlperf_conf=../../mlperf.conf \
    --user_conf=user.conf \
    --dataset-path=/datasets/mlperf-v0.5/dataset-imagenet-ilsvrc2012-val \
    --max-batchsize=1024 \
    --model=resnet50_v1.tflite \
    --model-name=resnet50 \
    --profile=resnet50-tf-ncore-offline \
    --scenario=Offline \
    --threads=2 \
    --output=/tmp/mlperf-tempout/output \
    --accuracy
```
