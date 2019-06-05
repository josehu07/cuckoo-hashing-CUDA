CC=g++
NVCC=nvcc
CXXFLAGS=-std=c++11 -O3

all: perf-test demo pre

perf-test: perf-test.cu cuckoo-cuda-naive.cuh cuckoo-cuda-multi.cuh
	${NVCC} $< -o $@ ${CXXFLAGS}

demo: demo.cu cuckoo-cuda-naive.cuh cuckoo-cuda-multi.cuh
	${NVCC} $< -o $@ ${CXXFLAGS}

pre: pre.cu cuckoo-cuda-naive.cuh cuckoo-cuda-multi.cuh
	${NVCC} $< -o $@ ${CXXFLAGS}

.PHONY: clean
clean:
	rm -f perf-test demo pre
