# Parallel Cuckoo Hashing on GPUs with CUDA

> Jose @ ShanghaiTech University
> 2019.05.31

![](https://img.shields.io/github/languages/count/hgz12345ssdlh/cuckoo-hashing-CUDA.svg?color=brightgreen)
![https://www.rust-lang.org/](https://img.shields.io/github/languages/top/hgz12345ssdlh/cuckoo-hashing-CUDA.svg)
![](https://img.shields.io/github/languages/code-size/hgz12345ssdlh/cuckoo-hashing-CUDA.svg)

## Description

Simple proof of concepts of parallel GPU hashing.

Under `Serial-vs-NaiveCUDA` is comparison between serial cuckoo hashing v.s. naive CUDA implementation of parallel cuckoo hashing. Under `NaiveCUDA-vs-Multilevel` is comparison between naive CUDA implementation v.s. multi-level shared-memory optimized parallel cuckoo hashing.

References:

- SIGGRAPH'09 paper: Dan A. Alcantara, Andrei Sharf, Fatemeh Abbasinejad, Shubhabrata Sengupta, Michael Mitzenmacher, John D. Owens, and Nina Amenta. 2009. Real-time parallel hashing on the GPU. ACM Trans. Graph. 28, 5, Article 154 (December 2009), 9 pages. DOI: [https://doi.org/10.1145/1618452.1618500](https://www.nvidia.com/content/GTC/posters/82_Alcantara_Real_Time_Parallel_Hashing.pdf). URL: [https://www.cs.bgu.ac.il/~asharf/Projects/RealTimeParallelHashingontheGPU.pdf](https://www.nvidia.com/content/GTC/posters/82_Alcantara_Real_Time_Parallel_Hashing.pdf)
- Poster of this paper: [https://www.nvidia.com/content/GTC/posters/82_Alcantara_Real_Time_Parallel_Hashing.pdf](https://www.nvidia.com/content/GTC/posters/82_Alcantara_Real_Time_Parallel_Hashing.pdf)

## How to Run

Environment prerequisites:

1. CUDA >= 10.1
2. C++ STL >= C++11 standard

Run the experiments:

- In each sub-directory, build by `make`
- Execute the built binaries (each described in its comments)
- Clean the build by `make clean`
