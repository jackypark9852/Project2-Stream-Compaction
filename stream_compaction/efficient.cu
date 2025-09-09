#include <cuda.h>
#include <cuda_runtime.h>
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
        /// performs upsweep part of work-efficient scan 
        /// </summary>
        /// <param name="n"> the number of element </param>
        /// <param name="d"> the iteration number </param>
        /// <param name="data"> the input/output buffer; the buffer is updated in place </param>
        /// <returns></returns>
        __global__ void kernUpSweep(int n, int d, int* data) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int stride = pow(2, d + 1); 
            int k = index * stride; 
            if (k >= n) {
                return;
            }
            
            data[k + stride - 1] += data[k + stride/2 - 1]; 
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // next power of two 
            int pot = pow(2, ilog2ceil(n));

            // initialize device side buffers 
            // buffers are padded so that its length is a power of two
            int* dev_data;
            size_t byteSize = pot * sizeof(int);
            cudaMalloc(&dev_data, byteSize);
            cudaMemset(dev_data, 0, byteSize);
            cudaMemcpy(dev_data + (pot - n), idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // launch config 
            int blockSize = 128;
            int blockCount = (pot + blockSize - 1) / blockSize;

            // start scan process
            timer().startGpuTimer();
            
            for (int d = 0; d <= ilog2ceil(n) - 1; ++d) {
                kernUpSweep<<<blockCount, blockSize>>>(pot, d, dev_data); 
            }

            timer().endGpuTimer();
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
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
