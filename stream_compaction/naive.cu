#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /// <summary>
        /// naive implementation of parallelized scan 
        /// </summary>
        /// <param name="n"> the number of elements; expects power of two </    param>
        /// <param name="d"> the current iteration number </param>
        /// <param name="odata"> the output buffer </param>
        /// <param name="idata"> the input buffer </param>
        /// <returns></returns>
        __global__ void kernNaiveScan(int n, int d, int* odata, const int* idata) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (k >= n) {
                return; 
            }
            
            int start = 1 << d; 
            odata[k] = (k >= start) ?
                (idata[k - start] + idata[k]) :
                idata[k];
        }

        __global__ void kernShiftRight(int n, int* odata, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }

            odata[index] = (index == 0) ? 0 : idata[index - 1]; 
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // next power of two 
            int pot = 1 << ilog2ceil(n); 
            
            // initialize device side buffers 
            // buffers are padded to the next power of two 
            int* dev_idata;
            int* dev_odata;
            size_t byteSize = pot * sizeof(int); 
            cudaMalloc(&dev_idata, byteSize);
            cudaMalloc(&dev_odata, byteSize);
            cudaMemset(dev_idata, 0, byteSize);
            cudaMemset(dev_odata, 0, byteSize);
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice); 

            // launch config 
            int blockSize = 8; 
            int blockCount = (n + blockSize - 1) / blockSize; 
            
            // run naive scan 
            timer().startGpuTimer();
            for (int d = 0; d < ilog2ceil(n); ++d) {
                kernNaiveScan << <blockCount, blockSize >> > (pot, d, dev_odata, dev_idata);
                std::swap(dev_odata, dev_idata); 
            }
            kernShiftRight << < blockCount, blockSize >> > (n, dev_odata, dev_idata);
            cudaDeviceSynchronize();
            timer().endGpuTimer();

            // retrieve scan result from device 
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // clean up resources 
            cudaFree(dev_idata); 
            cudaFree(dev_odata); 
        }
    }
}
