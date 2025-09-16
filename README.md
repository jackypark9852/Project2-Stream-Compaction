# CUDA Stream Compaction

## Project Introduction
> University of Pennsylvania **CIS5650 – GPU Programming and Architecture**  
> - Jacky Park  
> - Tested on: Windows 11, i9-13900H @ 2.60 32 GB, RTX 4070 (Laptop GPU) 8 GB (Personal Machine: ROG Zephyrus M16 GU604VI_GU604VI)

This project explores different implementations of **scan** (prefix sum) and **stream compaction** on both CPU and GPU.  

- **Scan**: an exclusive prefix sum, where each element becomes the sum of all previous values.  
- **Stream compaction**: removing unwanted elements, in this case filtering out `0`s.  

These are fundamental building blocks in parallel computing, and they show up everywhere from sorting to rendering pipelines. The project starts with a CPU reference, then moves through increasingly optimized GPU versions, and finally compares them to Thrust’s built-in implementation.

---

## Implementation Approaches

### 1. CPU Baseline

Functions:  
- `StreamCompaction::CPU::scan`  

The CPU version sets the **ground truth** for correctness.  

- **Scan** is just a `for` loop that accumulates into an output array. Simple, but it makes sure we have the right answers to compare against later.  

It’s still sequential, but it sets up the exact operations we’ll replicate in parallel on the GPU.


### 2. Naive GPU Scan

Function:  
- `StreamCompaction::Naive::scan`

This version follows the "naive" parallel scan algorithm described in [GPU Gems 3 (O39.2.1)](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).  

In each iteration, every element looks back a certain distance and adds its neighbor. First distance 1, then 2, then 4, doubling until the array is done. That means `ilog2ceil(n)` iterations, with one kernel launch per step.   The performance is better than the CPU for large sizes, but it’s still not very efficient.


### 3. Work-Efficient GPU Scan and Compaction

Functions:  
- `StreamCompaction::Efficient::scan`  
- `StreamCompaction::Efficient::compact`  
- `StreamCompaction::Common::kernMapToBoolean`  
- `StreamCompaction::Common::kernScatter`

This is where things get faster. The work-efficient scan is based on the Blelloch algorithm described in [GPU Gems 3 (O39.2.2)](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).  

#### Scan  
It runs in two phases:  
1. **Upsweep**: build a binary tree of partial sums.  
2. Set the last element to zero to prepare for exclusivity.  
3. **Downsweep**: walk back down the tree, writing prefix sums in place.  

The result is a scan that does only **O(n)** total work, instead of repeatedly touching the same values like the naive method. It also works in place with just one array.

#### Stream Compaction  
Compaction is built from three steps:  
1. Map the input to a boolean array (non-zero → 1, zero → 0).  
2. Scan that array.  
3. Scatter the non-zero values into their new compacted positions.  

On the GPU, all three steps are easy to parallelize, and the whole thing runs much faster than the CPU version (That is, if the array we're working on is large enough. More details in the test data later!). 

### 4. Thrust Implementation

Function:  
- `StreamCompaction::Thrust::scan`

The final version uses [NVIDIA’s Thrust library](https://developer.nvidia.com/thrust).  

It’s essentially a one-liner wrapping `thrust::exclusive_scan`, with device vectors handling the memory details. Thrust also provides `remove_if` for stream compaction, which achieves the same effect in a single call.  

This gives us a performance ceiling. The Thrust version is heavily optimized and tuned for real workloads, so our custom implementations can be judged against it.

---
## Results

### Block Size Exploration

Before diving into benchmarks for each scan implementation, I wanted to test how **block size** (threads per block) impacts performance.  

I expected block sizes **below 32** to perform poorly. Since GPU threads execute in groups of 32 (called a *warp*), a block size under 32 means some threads in every warp are left idle, wasting compute capacity. This exact issue showed up in my [CUDA Boids implementation](https://github.com/jackypark9852/Project1-CUDA-Flocking?tab=readme-ov-file#impact-of-block-size-on-performance).  

What I wasn’t sure about was how **larger block sizes** would behave. On Ada Lovelace GPUs, each SM can support up to **1536 threads concurrently**. A very large block size (like 1024) might actually reduce SM occupancy, since only one or two blocks could fit per SM at a time. That could limit how well the GPU hides latency.  

Speculation aside, here are the results. I ran each scan implementation on arrays of size **2^22**, repeating the test 10 times and reporting the average runtime in *milliseconds (ms)*.

<p align="center">
  <img src="img/block-size-perf.png" alt="Block size performance table"/>
</p>

<p align="center">
  <img src="img/block-size-perf-graph.png" alt="Block size performance graph"/>
</p>

Surprisingly, for both the **efficient scan** and the **Thrust scan**, changing the block size didn’t really matter. Their runtimes stayed consistent across the board.  

The **naive scan** told a different story. Performance improved steadily up to a block size of 64, after which things flattened out. Larger sizes, even as big as 1024, didn’t make a noticeable difference.  

Why is that? I’m not entirely sure yet. My guess is that the limiting factor isn’t occupancy, but something else in the kernel like memory bandwidth or synchronization. A deeper profiling session in **Nsight Compute** would probably reveal the real bottleneck.  

For now, a block size of **32 threads** looked like a solid baseline across all implementations. All subsequent benchmarks were run with this configuration.

---

