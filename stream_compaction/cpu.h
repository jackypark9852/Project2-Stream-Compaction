#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int* odata, const int* idata, bool startTimer = true);

        int compactWithoutScan(int n, int *odata, const int *idata);

        void map(int n, int* odata, const int* idata); 

        int scatter(int n, int* odata, const int* idata, const int* mdata, const int* sdata);

        int compactWithScan(int n, int *odata, const int *idata);
    }
}
