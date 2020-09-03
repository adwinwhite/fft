#include<iostream>
#include<vector>
#include<tuple>
#include<complex>
#include<numeric>
#include<cmath>
#include<memory>
#include<bitset>
#include<chrono>
#include<cstdlib>
#include"matplotlibcpp.h"


const double PI = std::acos(-1.0);

std::complex<double> oddFactor(const unsigned numOfPoints, const unsigned inputIndex, const bool isInverse) {
    using namespace std::complex_literals;
    return std::exp((isInverse ? 1.0 : -1.0) * 2i * PI * double(inputIndex) / double(numOfPoints));
}


const unsigned BIT_REVERSE_TALBE8 [] = {
    0,  128, 64, 192, 32, 160,  96, 224, 16, 144, 80, 208, 48, 176, 112, 240,
    8,  136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248,
    4,  132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244,
    12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252,
    2,  130, 66, 194, 34, 162,  98, 226, 18, 146, 82, 210, 50, 178, 114, 242,
    10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250,
    6,  134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246,
    14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254,
    1,  129, 65, 193, 33, 161,  97, 225, 17, 145, 81, 209, 49, 177, 113, 241,
    9,  137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249,
    5,  133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245,
    13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253,
    3,  131, 67, 195, 35, 163,  99, 227, 19, 147, 83, 211, 51, 179, 115, 243,
    11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251,
    7,  135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247,
    15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255
};

unsigned bitReverse32(const unsigned orig) {
    unsigned result = 0;
    for (unsigned i = 0; i < 4; ++i) {
        result = result | (BIT_REVERSE_TALBE8[((orig << (i * 8)) >> 24)] << (i * 8));
    }
    return result;
}

unsigned bitReverseInt(const unsigned orig, const unsigned numOfBits) {
    if (numOfBits > 32) {
        std::cerr << "The size of bits to be reversed should be smaller than 32" << std::endl;
        std::exit(-1);
    }
    return bitReverse32(orig) >> (32 - numOfBits);
}

/*  The size should be smaller than 2^32.Otherwise, the code needs modifying.
 *  The size should be power of 2.
 */
std::vector<std::complex<double>> cfft(std::vector<std::complex<double>> samples, const bool isInverse) {
    const unsigned sampleSize = unsigned(samples.size());
    std::vector<std::complex<double>> samplesBuffer(sampleSize);
    const unsigned numOfBits = unsigned(std::log2(sampleSize));

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 4)
        for (unsigned i = 0; i < sampleSize; i++) {
            samplesBuffer[i] = samples[bitReverseInt(i, numOfBits)];
        }
    }

    auto currentBuffer = std::unique_ptr<std::vector<std::complex<double>>>(&samplesBuffer);
    auto nextBuffer = std::unique_ptr<std::vector<std::complex<double>>>(&samples);


    for (unsigned i = 2; i <= sampleSize; i *= 2) {
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 4)
            for (unsigned j = 0; j < sampleSize; ++j) {
                (*nextBuffer)[j] = j % i < i / 2 ? (*currentBuffer)[j] + oddFactor(i, j, isInverse) * (*currentBuffer)[j + i / 2] : oddFactor(i, j, isInverse) * (*currentBuffer)[j] + (*currentBuffer)[j - i / 2];
            }
        }
        currentBuffer.swap(nextBuffer);
    }


    if (isInverse) {
        //"parallel for" is not compatible with "for (auto x : xs)".
        #pragma omp parallel for
        for(unsigned i = 0; i < sampleSize; ++i) {
            (*nextBuffer)[i] /= double(sampleSize);
        }
    }

    // It's necessary to release the ownership. Otherwise the returned value can't be assigned to another variable.
    currentBuffer.release();
    if (nextBuffer.get() == &samples) {
        nextBuffer.release();
        return samples;
    } else {
        nextBuffer.release();
        return samplesBuffer;
    }
    //return *nextBuffer;
}

std::complex<double> preBesselF(const std::complex<double> z, const unsigned numOfPoints, const unsigned inputIndex) {
    using namespace std::complex_literals;
    return std::exp(1i * z * std::cos(2 * PI * inputIndex / numOfPoints));
}

std::vector<std::complex<double>> bessel(const std::complex<double> z, const unsigned order) {
    if ((order & (order - 1)) != 0) {
        std::cerr << "order must be power of 2" << std::endl;
        std::exit(-1);
    }
    auto xs = std::vector<std::complex<double>>(order);
    for (unsigned i = 0; i< xs.capacity(); ++i) {
        xs[i] = preBesselF(z, order, i);
    }
    auto Xs = cfft(xs, true);
    using namespace std::complex_literals;
    for (unsigned l = 0; l < order; ++l) {
        Xs[l] *= std::pow(1i, -l) * 2.0;
    }
    return Xs;
}

//results are more accurate than homework1.
int main()
{
    namespace plt = matplotlibcpp;
    const unsigned ORDER = unsigned(std::pow(2, 16));
    std::vector<double> xs(ORDER);
    std::iota(xs.begin(), xs.end(), 0);
    for (unsigned i = 0; i < 3; ++i) {
        auto ys = bessel(std::complex<double>(std::pow(10, i), 0), ORDER);
        auto ysr = std::vector<double>(ORDER);
        for (unsigned j = 0; j < ORDER; ++j) {
            ysr[j] = ys[j].real();
        }
        plt::named_plot(std::to_string(std::pow(10, i)), xs, ysr);
        std::cout << "First ten terms of J(" << std::pow(10, i) << ", n)" << std::endl;
        std::cout << "n : J" << std::endl;
        for (unsigned j = 0; j < 10; ++j) {
            std::cout << j << " : " << ys[j] << std::endl;
        }
    }
    plt::legend();
    plt::xlim(0, 32);
    plt::show();
    return 0;
}
