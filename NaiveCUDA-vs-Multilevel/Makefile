CC=g++
NVCC=nvcc
CXXFLAGS=-std=c++11 -O3

all: perf demo test

perf: perf.cu cuckoo-cuda-naive.cuh cuckoo-cuda-multi.cuh
	${NVCC} $< -o $@ ${CXXFLAGS}

demo: demo.cu cuckoo-cuda-naive.cuh cuckoo-cuda-multi.cuh
	${NVCC} $< -o $@ ${CXXFLAGS}

test: test.cu cuckoo-cuda-naive.cuh cuckoo-cuda-multi.cuh
	${NVCC} $< -o $@ ${CXXFLAGS}

.PHONY: clean
clean:
	rm -f perf demo test
