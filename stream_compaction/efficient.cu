#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /// <summary>
        /// performs the upsweep part of work-efficient scan 
        /// </summary>
        /// <param name="n"> the number of element </param>
        /// <param name="d"> the iteration number </param>
        /// <param name="data"> the input/output buffer; the buffer is updated in place </param>
        /// <returns></returns>
        __global__ void kernUpSweep(int n, int d, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;
            int stride = 1 << (d + 1);
            int half = 1 << d;

            int groups = n / stride;
            if (index >= groups) return;

            int k = index * stride;
            if (k + stride - 1 >= n) {
                return;
            }

            data[k + stride - 1] += data[k + half - 1];
        }

        /// <summary>
        /// performd the downsweep part of work-efficient scan 
        /// </summary>
        /// <param name="n"></param>
        /// <param name="d"></param>
        /// <param name="data"></param>
        /// <returns></returns>
        __global__ void kernDownSweep(int n, int d, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;
            int stride = 1 << (d + 1);
            int half = 1 << d;

            int groups = n / stride;
            if (index >= groups) return;

            int k = index * stride;
            if (k + stride - 1 >= n) {
                return;
            }

            int temp = data[k + half - 1]; // stride/2 is pow(2, d) 
            data[k + half - 1] = data[k + stride - 1];
            data[k + stride - 1] += temp;
        }

        __global__ void kernMapToBoolean(int n, int* mdata, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            mdata[index] = idata[index] != 0;
        }

        __global__ void kernScatter(int n, int* odata,
            const int* sdata,
            const int* idata) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;
            if (idata[i]) {
                odata[sdata[i]] = idata[i];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata, bool startTimer, bool usingDevicePtr) {
            // next power of two 
            int pot = 1 << ilog2ceil(n);

            // initialize device side buffers 
            // buffers are padded so that its length is a power of two
            int* dev_data;
            size_t byteSize = pot * sizeof(int);
            cudaMalloc(&dev_data, byteSize);
            cudaMemset(dev_data, 0, byteSize);
            auto memCpyMode = (usingDevicePtr) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
            cudaMemcpy(dev_data, idata, n * sizeof(int), memCpyMode);

            // launch config 
            int blockSize = 8;
            // start scan process
            int exponent = ilog2ceil(n);
            int d; 
            if (startTimer) timer().startGpuTimer();
            for (d = 0; d <= exponent - 1; ++d) {
                // Launch only the necessary number of blocks
                int stride = 1 << (d + 1);
                int groups = pot / stride;
                int blockCount = (groups + blockSize - 1) / blockSize;
                kernUpSweep << <blockCount, blockSize >> > (pot, d, dev_data);
            }

            // prep for downsweep by setting last element to 0 
            cudaMemset(dev_data + (pot - 1), 0, sizeof(int));

            for (d = exponent - 1; d >= 0; --d) {
                int stride = 1 << (d + 1);
                int groups = pot / stride;
                int blockCount = (groups + blockSize - 1) / blockSize;
                kernDownSweep << <blockCount, blockSize >> > (pot, d, dev_data);
            }
            cudaDeviceSynchronize();
            if (startTimer) timer().endGpuTimer();

            // fetch result from device 
            memCpyMode = (usingDevicePtr) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
            cudaMemcpy(odata, dev_data, n * sizeof(int), memCpyMode);
            cudaFree(dev_data); 
        }



        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            int* dev_idata = nullptr, * dev_flags = nullptr, * dev_index = nullptr, * dev_out = nullptr;
            cudaMalloc(&dev_idata, n * sizeof(int));
            cudaMalloc(&dev_flags, n * sizeof(int));
            cudaMalloc(&dev_index, n * sizeof(int));
            cudaMalloc(&dev_out, n * sizeof(int));  // worst-case size

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 8;
            int blockCount = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            kernMapToBoolean << <blockCount, blockSize >> > (n, dev_flags, dev_idata);

            scan(n, dev_index, dev_flags, /*startTimer=*/false, /*usingDevicePtr=*/true);
            
            kernScatter << <blockCount, blockSize >> > (n, dev_out, dev_index, dev_idata);
            cudaDeviceSynchronize();
            timer().endGpuTimer();

            int lastScan = 0, lastVal = 0;
            cudaMemcpy(&lastScan, dev_index + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastVal, dev_idata + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastScan + (lastVal != 0);

            cudaMemcpy(odata, dev_out, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_out);
            cudaFree(dev_index);
            cudaFree(dev_flags);
            cudaFree(dev_idata);

            return count;
        }
    }
    }