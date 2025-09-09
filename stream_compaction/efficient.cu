#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
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
            int stride = pow(2, d + 1); 
            int k = index * stride; 
            if (k >= n) {
                return;
            }
            
            data[k + stride - 1] += data[k + stride/2 - 1]; 
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
            int stride = pow(2, d + 1);
            int k = index * stride;
            if (k >= n) {
                return;
            }
            
            int temp = data[k + stride/2 - 1]; // stride/2 is pow(2, d) 
            data[k + stride / 2 - 1] = data[k + stride - 1]; 
            data[k + stride - 1] += temp; 
        }

        __global__ void kernMapToBoolean(int n, int* mdata, int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return; 
            }

            mdata[index] = idata[index] != 0; 
        }
        
        __global__ void  kernScatter(int n ,int* odata, const int* sdata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
       
            if (idata[index] ) {
                odata[sdata[index]] = idata[index]; 
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool startTimer, bool usingDevicePtr) {
            // next power of two 
            int pot = pow(2, ilog2ceil(n));

            // initialize device side buffers 
            // buffers are padded so that its length is a power of two
            int* dev_data;
            size_t byteSize = pot * sizeof(int);
            cudaMalloc(&dev_data, byteSize);
            cudaMemset(dev_data, 0, byteSize);
            auto memCpyMode = (usingDevicePtr) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice; 
            cudaMemcpy(dev_data + (pot - n), idata, n * sizeof(int), memCpyMode);

            // launch config 
            int blockSize = 128;
            int blockCount = (pot + blockSize - 1) / blockSize;

            // start scan process
            if(startTimer) timer().startGpuTimer();
            for (int d = 0; d <= ilog2ceil(n) - 1; ++d) {
                kernUpSweep << <blockCount, blockSize >> > (pot, d, dev_data);
            }

            // prep for downsweep by setting last element to 0 
            cudaMemset(dev_data + (pot - 1), 0, sizeof(int));

            for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
                kernDownSweep << <blockCount, blockSize >> > (pot, d, dev_data);
            }
            if(startTimer) timer().endGpuTimer();

            // fetch result from device 
            memCpyMode = (usingDevicePtr) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost; 
            cudaMemcpy(odata, dev_data + (pot - n), n * sizeof(int), memCpyMode);
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
            // initialize device side buffers 
            int* dev_odata; 
            int* dev_idata; 
            cudaMalloc(&dev_odata, n * sizeof(int)); 
            cudaMalloc(&dev_idata, n * sizeof(int)); 
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice); 

            // launch config 
            int blockSize = 128;
            int blockCount = (n + blockSize - 1) / blockSize; 

            // convert input buf to booleans 
            kernMapToBoolean <<< blockCount, blockSize >> > (n, dev_odata, dev_idata); 

            // scan to store indices used in compaction
            scan(n, dev_odata, dev_odata, false, true);
            int lastScan = 0;
            int lastValue = 0;

            cudaMemcpy(&lastScan, dev_odata + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastValue, dev_idata + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            int lastFlag = (lastValue != 0);
            int count = lastScan + lastFlag;


            // After cudaMemcpy
            std::vector<int> h_debug(n);
            cudaMemcpy(h_debug.data(), dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Print
            printf("dev_data contents after memcpy:\n");
            for (int i = 0; i < n; i++) {
                printf("%d ", h_debug[i]);
            }
            printf("\n");

            // compact 
            kernScatter << < blockCount, blockSize >> > (n, dev_odata, dev_odata, dev_idata); 
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost); 

            // TODO
            timer().endGpuTimer();
            return count;
        }
    }
}
