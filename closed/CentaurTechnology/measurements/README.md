# Running Docker Container

To run benchmarks on Centaur Technology's Ncore accelerator, use the provided docker container `closed/CentaurTechnology/measurements/base.Dockerfile`.

Build and run the docker container. Replace `path-to-mlperf-datasets` with your specific path to the input datasets used by the MLPerf benchmark suite. Replace `path-to-custom-workspace` with the path to the `closed/CentaurTechnology/code/` path provided.

```
docker image build -f base.Dockerfile -t ncore/base/develop:latest .
docker container run --rm                        \
    --device /dev/ncore_pci                      \
    --volume /path-to-custom-workspace/:/workspace \
    --volume /path-to-mlperf-datasets/:/datasets \
    -it ncore/base/develop:latest /bin/bash
```
