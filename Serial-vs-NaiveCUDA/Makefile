CC=g++
NVCC=nvcc
CXXFLAGS=-std=c++11 -O3

all: perf-test demo

perf-test: perf-test.cu cuckoo-serial.hpp cuckoo-cuda.cuh
	${NVCC} $< -o $@ ${CXXFLAGS}

demo: demo.cu cuckoo-serial.hpp cuckoo-cuda.cuh
	${NVCC} $< -o $@ ${CXXFLAGS}

.PHONY: clean
clean:
	rm -f demo perf-test
