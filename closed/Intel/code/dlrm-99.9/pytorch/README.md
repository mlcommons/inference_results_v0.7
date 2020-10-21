# DLRM MLPerf Inference v0.7 Intel Submission

## HW and SW requirements
### 1. HW requirements
| HW | configuration |
| -: | :- |
| CPU | CPX-6 @ 8 sockets/Node |
| DDR | 192G/socket @ 3200 MT/s |
| SSD | 1 SSD/Node @ >= 1T |

### 2. SW requirements
| SW |configuration  |
|--|--|
| GCC | GCC 9.3  |

## Steps to run DLRM

### 1. Install anaconda 3.0
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh                                                         
  ~/anaconda3.sh -b -p ~/anaconda3                                              
  ~/anaconda3/bin/conda create -n dlrm python=3.7                               
                                                                                
  export PATH=~/anaconda3/bin:$PATH                                             
  source ~/anaconda3/bin/activate dlrm
```
### 2. Install dependency packages
```
  pip install sklearn onnx tqdm lark-parser                                     
  pip install -e git+https://github.com/mlperf/logging@0.7.0-rc2#egg=logging       
  conda install ninja pyyaml setuptools cmake cffi typing                       
  conda install intel-openmp mkl mkl-include numpy --no-update-deps             
  conda install -c conda-forge gperftools                                       
  pip install opencv-python pybind11 absl-py opencv-python-headless
```
### 3. Clone source code and Install
(1) Install PyTorch and Intel Extension for PyTorch
```
   # clone PyTorch
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch && git checkout tags/v1.5.0-rc3 -b v1.5-rc3
   git submodule sync && git submodule update --init --recursive

   # clone Intel Extension for PyTorch
   git clone https://github.com/intel/intel-extension-for-pytorch.git
   cd intel-extension-for-pytorch
   git checkout 0.2
   git submodule update --init --recursive

   # install PyTorch
   cd {path/to/pytorch}                                                          
   cp {path/to/intel-pytorch-extension}/torch_patches/0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch .
   patch -p1 < 0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch         
   python setup.py install

   # install Intel Extension for PyTorch
   cd {path/to/intel-pytorch-extension}
   python setup.py install
```
(2) Install loadgen
```
   git clone https://github.com/mlperf/inference.git
   cd inference && git checkout dd9e6bf
   cd loadgen && CFLAGS="-std=c++14" python setup.py install
```
(3) Prepare source code
```
   git clone /path/to/this/git/repo mlperf
   cd mlperf/closed/Intel/code/dlrm/pytorch
```
### 4. Run command for server and offline mode
(1) configurable options
```
   export NUM_SOCKETS=        # i.e. 8
   export CPUS_PER_SOCKET=    # i.e. 28
   export CPUS_PER_INSTANCE=  # i.e. 14. Used by server mode
   export DATA_DIR=           # the path of dlrm dataset
   export MODEL_DIR=          # the path of dlrm pre-trained model
```
(2) command line
```
   # server-performance-mode
   sudo ./run_clean.sh
   ./run_main.sh server
   
   # server-accuracy-mode
   sudo ./run_clean.sh
   ./run_main.sh server accuracy
   
   # offline-performance-mode
   sudo ./run_clean.sh
   ./run_main.sh offline
   
   # offline-accuracy-mode
   sudo ./run_clean.sh
   ./run_main.sh offline accuracy
   
```
