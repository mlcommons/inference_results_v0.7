# MLPerf Inference v0.7 - OpenVINO

## SSD-ResNet34

### SingleStream

#### Native

- Set up the [OpenVINO program wrapper](https://github.com/dividiti/ck-openvino-private/blob/master/program/openvino-loadgen-v0.7-drop/README.setup.md) on your SUT.
- Customize the examples below for your SUT.

##### Performance

```bash
$ ck run cmdgen:benchmark.openvino-loadgen --verbose \
--sut=dellemc_r640xd6248r --model=ssd-resnet34 --division=open \
--scenario=singlestream --mode=performance --target_latency=50 \
--warmup_iters=100 --nthreads=24 --nireq=1 --nstreams=1
```

##### Accuracy

```bash
$ ck run cmdgen:benchmark.openvino-loadgen --verbose \
--sut=dellemc_r640xd6248r --model=ssd-resnet34 --division=open \
--scenario=singlestream --mode=accuracy --target_latency=50 \
--warmup_iters=10 --nthreads=24 --nireq=1 --nstreams=1
...
mAP=19.928%
```

##### Compliance

```bash
$ ck run cmdgen:benchmark.openvino-loadgen --verbose \
--sut=dellemc_r640xd6248r --model=ssd-resnet34 --division=open \
--scenario=singlestream --compliance,=TEST01,TEST04-A,TEST04-B,TEST05 --target_latency=50 \
--warmup_iters=100 --nthreads=24 --nireq=1 --nstreams=1
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
--user=$(id -u):1500 --rm ${CK_ORG}/${CK_IMAGE}:${CK_TAG} \
"date && time ck run cmdgen:benchmark.openvino-loadgen --verbose \
--sut=aws-g4dn.4xlarge --model=ssd-resnet34 --division=open \
--scenario=singlestream --mode=performance --target_latency=100 \
--warmup_iters=100 --nthreads=16 --nireq=1 --nstreams=1 && date"
```

##### Accuracy

```bash
$ docker run --env-file ${CK_REPOS}/ck-openvino-private/docker/${CK_IMAGE}/env.list \
--volume ${CK_OPENVINO_PRIVATE_DIR}:/home/dvdt/CK_REPOS/${CK_OPENVINO_PRIVATE_REPO} \
--volume ${CK_EXPERIMENTS_DIR}:/home/dvdt/CK_REPOS/local/experiment \
--volume ${CK_OPENVINO_DROP}:/home/dvdt/drop \
--user=$(id -u):1500 --rm ${CK_ORG}/${CK_IMAGE}:${CK_TAG} \
"date && time ck run cmdgen:benchmark.openvino-loadgen --verbose \
--sut=aws-g4dn.4xlarge --model=ssd-resnet34 --division=open \
--scenario=singlestream --mode=accuracy --target_latency=100 \
--warmup_iters=10 --nthreads=16 --nireq=1 --nstreams=1 && date"
...
mAP=19.928%
```

##### Compliance

```bash
$ docker run --env-file ${CK_REPOS}/ck-openvino-private/docker/${CK_IMAGE}/env.list \
--volume ${CK_OPENVINO_PRIVATE_DIR}:/home/dvdt/CK_REPOS/${CK_OPENVINO_PRIVATE_REPO} \
--volume ${CK_EXPERIMENTS_DIR}:/home/dvdt/CK_REPOS/local/experiment \
--volume ${CK_OPENVINO_DROP}:/home/dvdt/drop \
--user=$(id -u):1500 --rm ${CK_ORG}/${CK_IMAGE}:${CK_TAG} \
"date && time ck run cmdgen:benchmark.openvino-loadgen --verbose \
--sut=aws-g4dn.4xlarge --model=ssd-resnet34 --division=open \
--scenario=singlestream --compliance,=TEST01,TEST04-A,TEST04-B,TEST05 --target_latency=100 \
--warmup_iters=100 --nthreads=16 --nireq=1 --nstreams=1 && date"
```
