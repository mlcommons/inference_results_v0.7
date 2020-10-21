PROJECT   =   app.exe

CXX       :=   g++

VAI_ALVEO_ROOT=/usr/include
#MLPERF_INFERENCE=/proj/rdi/staff/gaoyue/gaoyue/mlperf_inference
#MLPERF_INFERENCE=/scratch/mlperf/mlperf_inference

#LDFLAGS   :=     -L/usr/local/lib64  -L $(MLPERF_INFERENCE)/loadgen  -L $(CONDA_PREFIX)/lib -L/lib64
#LDFLAGS   :=     -L/usr/local/lib64  -L $(MLPERF_INFERENCE)/loadgen  -L $(CONDA_PREFIX)/lib -L/usr/local/protobuf/lib
#LDFLAGS   :=     -L/wrk/released/2019.2/XRT/2019.2/201920.2.3.1301/packages/xrt-2.1.0-centos/opt/xilinx/xrt/lib -L$(PWD) -L$(PWD)/../DNNDK/n2cube/build -L/usr/local/lib64  -L $(MLPERF_INFERENCE)/loadgen  -L $(CONDA_PREFIX)/lib -L/usr/local/lib 
LDFLAGS   :=     -L/opt/xilinx/xrt/lib -L$(PWD) -L$(PWD)/../DNNDK/n2cube/build -L/usr/local/lib64  -L $(MLPERF_INFERENCE)/loadgen   -L/usr/local/lib 
#LDFLAGS   :=     -L$(PWD) -L$(PWD)/../DNNDK/n2cube/build -L/usr/local/lib64  -L $(MLPERF_INFERENCE)/loadgen  -L $(CONDA_PREFIX)/lib -L/usr/local/lib 
LDFLAGS   +=     -luuid -lxilinxopencl -lxrt_core -lmlperf_loadgen -lpthread -lopencv_imgcodecs -lopencv_core -lopencv_highgui -lopencv_imgproc   -lpthread -ln2cube -lboost_system -lprotobuf -lglog



CUR_DIR =   $(shell pwd)
SRC     =   $(CUR_DIR)
SRC     +=   $(CUR_DIR)/tfssd
SRC     +=   $(CUR_DIR)/tfssd/anchor
SRC     +=   $(CUR_DIR)/model_config/src
SRC     +=   $(CUR_DIR)/include/object_detection/protos
SRC     +=   $(CUR_DIR)/model_config/include/vitis/ai/proto
SRC     +=   $(CUR_DIR)/math/src
BUILD   =   $(CUR_DIR)/build
VPATH   =   $(SRC)
C_DIR  :=   $(shell find $(SRC) -name *.c)
OBJ     =   $(patsubst %.c, %.o, $(notdir $(C_DIR)))
CC_DIR :=   $(shell find $(SRC) -name *.cc)
OBJ    +=   $(patsubst %.cc, %.o, $(notdir $(CC_DIR)))
CPP_DIR :=   $(shell find $(SRC) -name *.cpp)
OBJ    +=   $(patsubst %.cpp, %.o, $(notdir $(CPP_DIR)))
PB_DIR :=   $(shell find $(SRC) -name *.pb.cc)
OBJ    +=   $(patsubst %.pb.cc, %.pb.o, $(notdir $(PB_DIR)))

CFLAGS :=  -D ENABLE_INT  -I$(VAI_ALVEO_ROOT)/vitis -I$(MLPERF_INFERENCE)/loadgen -I$(CONDA_PREFIX)/include -O3 
#CFLAGS :=  -D ENABLE_INT  -I$(VAI_ALVEO_ROOT)/vitis -I$(MLPERF_INFERENCE)/loadgen -I$(CONDA_PREFIX)/include -g 
CFLAGS +=   -I$(CUR_DIR)/include  -I$(CUR_DIR)/model_config/include -I$(CUR_DIR)/math/include -I/usr/local/include/google/protobuf  -I$(CUR_DIR)/include/object_detection/protos/
#CFLAGS +=   -I$(CUR_DIR)/include  -I$(CUR_DIR)/model_config/include 

CFLAG += -DTGT_DEVICE=${TARGET} 

.PHONY: all clean

all: $(BUILD) $(PROJECT)

$(PROJECT) : $(OBJ)
	$(CXX) $(CFLAGS) $(addprefix $(BUILD)/, $^) $(MODEL) -o $@ $(LDFLAGS)

%.o : %.cc
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@
%.o : %.cpp
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@

clean:
	$(RM) -rf $(BUILD)/*.o $(BUILD)
	$(RM) $(PROJECT)

$(BUILD) :
	-mkdir -p $@
	cd include; ./build.sh
	cd model_config/include/vitis/ai/proto; protoc  --cpp_out=./ --proto_path=./ dpu_model_param.proto

