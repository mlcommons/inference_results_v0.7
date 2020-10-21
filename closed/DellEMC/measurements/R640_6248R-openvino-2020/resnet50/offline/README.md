# MLPerf Inference v0.7 - OpenVINO

## ResNet50

### Offline

#### Native

- Set up the [OpenVINO program wrapper](https://github.com/dividiti/ck-openvino-private/blob/master/program/openvino-loadgen-v0.7-drop/README.setup.md) on your SUT.
- Customize the examples below for your SUT.

##### Performance

```bash
$ ck run cmdgen:benchmark.openvino-loadgen --verbose \
--sut=dellemc_r640xd6248r --model=resnet50 \
--scenario=offline --mode=performance --target_qps=2500 \
--warmup_iters=1000 --nvirtcpus=`grep -c processor /proc/cpuinfo` \
--nthreads={{{nvirtcpus}}} --nireq={{{nvirtcpus}}} --nstreams={{{nvirtcpus}}}
```

##### Accuracy

```bash
$ ck run cmdgen:benchmark.openvino-loadgen --verbose \
--sut=dellemc_r640xd6248r --model=resnet50 \
--scenario=offline --mode=accuracy --target_qps=2500 \
--warmup_iters=100 --nvirtcpus=`grep -c processor /proc/cpuinfo` \
--nthreads={{{nvirtcpus}}} --nireq={{{nvirtcpus}}} --nstreams={{{nvirtcpus}}}
...
accuracy=76.292%, good=38146, total=50000
```

##### Compliance

```bash
$ ck run cmdgen:benchmark.openvino-loadgen --verbose \
--sut=dellemc_r640xd6248r --model=resnet50 \
--scenario=offline --compliance,=TEST01,TEST04-A,TEST04-B,TEST05 --target_qps=2500 \
--warmup_iters=100 --nvirtcpus=`grep -c processor /proc/cpuinfo` \
--nthreads={{{nvirtcpus}}} --nireq={{{nvirtcpus}}} --nstreams={{{nvirtcpus}}}
```

#### Docker

- Set up the [OpenVINO Docker image](https://github.com/dividiti/ck-openvino-private/blob/master/docker/openvino-loadgen-v0.7-drop/README.md) on your SUT.
- Customize the examples below for your SUT.

##### Performance

```bash
$ docker run --env-file ${CK_REPOS}/ck-openvino-private/docker/${CK_IMAGE}/env.list \
--volume ${CK_OPENVINO_PRIVATE_DIR}:/home/dvdt/CK_REPOS/${CK_OPENVINO_PRIVATE_REPO} \
--volume ${CK_EXPERIMENTS_DIR}:/home/dvdt/CK_REPOS/local/experiment \
--volume ${CK_OPENVINO_DROP}:/home/dvdt/drop \
--volume ${CK_DATASET_IMAGENET}:/home/dvdt/imagenet \
--user=$(id -u):1500 --rm ${CK_ORG}/${CK_IMAGE}:${CK_TAG} \
"ck detect soft:dataset.imagenet.val --full_path=/home/dvdt/imagenet/ILSVRC2012_val_00000001.JPEG \
&& cp /home/dvdt/CK_TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt /home/dvdt/imagenet/val_map.txt \
&& date && time ck run cmdgen:benchmark.openvino-loadgen --verbose \
--sut=aws-g4dn.4xlarge --model=resnet50 \
--scenario=offline --mode=performance --target_qps=500 \
--warmup_iters=1000 --nvirtcpus=`grep -c processor /proc/cpuinfo` \
--nthreads={{{nvirtcpus}}} --nireq={{{nvirtcpus}}} --nstreams={{{nvirtcpus}}} && date"
```

##### Accuracy

```bash
$ docker run --env-file ${CK_REPOS}/ck-openvino-private/docker/${CK_IMAGE}/env.list \
--volume ${CK_OPENVINO_PRIVATE_DIR}:/home/dvdt/CK_REPOS/${CK_OPENVINO_PRIVATE_REPO} \
--volume ${CK_EXPERIMENTS_DIR}:/home/dvdt/CK_REPOS/local/experiment \
--volume ${CK_OPENVINO_DROP}:/home/dvdt/drop \
--volume ${CK_DATASET_IMAGENET}:/home/dvdt/imagenet \
--user=$(id -u):1500 --rm ${CK_ORG}/${CK_IMAGE}:${CK_TAG} \
"ck detect soft:dataset.imagenet.val --full_path=/home/dvdt/imagenet/ILSVRC2012_val_00000001.JPEG \
&& cp /home/dvdt/CK_TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt /home/dvdt/imagenet/val_map.txt \
&& date && time ck run cmdgen:benchmark.openvino-loadgen --verbose \
--sut=aws-g4dn.4xlarge --model=resnet50 \
--scenario=offline --mode=accuracy --target_qps=500 \
--warmup_iters=100 --nvirtcpus=`grep -c processor /proc/cpuinfo` \
--nthreads={{{nvirtcpus}}} --nireq={{{nvirtcpus}}} --nstreams={{{nvirtcpus}}} && date"
...
accuracy=76.292%, good=38146, total=50000
```

##### Compliance

```bash
$ docker run --env-file ${CK_REPOS}/ck-openvino-private/docker/${CK_IMAGE}/env.list \
--volume ${CK_OPENVINO_PRIVATE_DIR}:/home/dvdt/CK_REPOS/${CK_OPENVINO_PRIVATE_REPO} \
--volume ${CK_EXPERIMENTS_DIR}:/home/dvdt/CK_REPOS/local/experiment \
--volume ${CK_OPENVINO_DROP}:/home/dvdt/drop \
--volume ${CK_DATASET_IMAGENET}:/home/dvdt/imagenet \
--user=$(id -u):1500 --rm ${CK_ORG}/${CK_IMAGE}:${CK_TAG} \
"ck detect soft:dataset.imagenet.val --full_path=/home/dvdt/imagenet/ILSVRC2012_val_00000001.JPEG \
&& cp /home/dvdt/CK_TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt /home/dvdt/imagenet/val_map.txt \
&& date && time ck run cmdgen:benchmark.openvino-loadgen --verbose \
--sut=aws-g4dn.4xlarge --model=resnet50 \
--scenario=offline --compliance,=TEST01,TEST04-A,TEST04-B,TEST05 --target_qps=500 \
--warmup_iters=1000 --nvirtcpus=`grep -c processor /proc/cpuinfo` \
--nthreads={{{nvirtcpus}}} --nireq={{{nvirtcpus}}} --nstreams={{{nvirtcpus}}} && date"
```
