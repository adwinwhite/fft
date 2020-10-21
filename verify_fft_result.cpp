#include <iostream>
#include <cmath>
#include <fstream>
#include "fft_uniprocessor.h"

//#include <omp.h>


int main() {
    constexpr double start = 0;
    constexpr double step = 0.1;
    constexpr size_t num = 8;
    auto samples = new myfft::mcomplex[num];
    for (size_t i = 0; i < num; ++i) {
        samples[i][0] = std::sin(i * step + start);
        samples[i][1] = 0;
    }
    auto results = myfft::cfft_c(samples, num);
    for (size_t i = 0; i < num; ++i) {
        std::cout << i << " : " << results[i][0] << ", " << results[i][1] << std::endl;
    }
    return 0;
}
