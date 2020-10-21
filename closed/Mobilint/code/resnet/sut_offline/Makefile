CXX := g++
CXXFLAGS := -pthread -O3 -I. -fopenmp -ffast-math
LDFLAGS := -Llib -lmaccel

SRCS := $(wildcard *.cc)

all: benchmark

benchmark: $(SRCS) lib/libmlperf_loadgen.a
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	@echo "Sweeping all the artifacts..."
	@rm -rf benchmark