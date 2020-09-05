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
#include<fftw3.h>
#include<omp.h>

const double EPSILON = std::pow(2, -52);
const double PI = std::acos(-1.0);

std::complex<double> oddFactor(const unsigned& numOfPoints, const unsigned& inputIndex, const bool& isInverse) {
    using namespace std::complex_literals;
    return std::exp((isInverse ? 1.0 : -1.0) * 2i * PI * double(inputIndex) / double(numOfPoints));
}

std::complex<double> oddFactor(const unsigned& numOfPoints, const unsigned& inputIndex, const double& inverseFactor = -1.0) {
    using namespace std::complex_literals;
    return std::exp(inverseFactor * 2i * PI * double(inputIndex) / double(numOfPoints));
}

std::complex<double> oddFactor(const double& numOfPointsReciprocal, const unsigned& inputIndex, const bool& isInverse) {
    using namespace std::complex_literals;
    return std::exp((isInverse ? 1.0 : -1.0) * 2i * PI * double(inputIndex) * numOfPointsReciprocal);
}

std::complex<double> oddFactor(const double& numOfPointsReciprocal, const unsigned& inputIndex, const double& inverseFactor = -1.0) {
    using namespace std::complex_literals;
    return std::exp(inverseFactor * 2i * PI * double(inputIndex) * numOfPointsReciprocal);
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

unsigned bitReversePerByte(const unsigned& orig, const unsigned& numOfBytes) {
    unsigned result = 0;
    for (unsigned i = 0; i < numOfBytes; ++i) {
        result = result | (BIT_REVERSE_TALBE8[((orig << (i * 8)) >> (8 * (numOfBytes - 1)))] << (i * 8));
    }
    return result;
}

//compatible with size_t now
unsigned bitReverseInt(const unsigned& orig, const unsigned& numOfBits) {
    if (numOfBits > 8 * sizeof(unsigned)) {
        std::cerr << "The size of bits to be reversed should be smaller than unsigned's" << std::endl;
        std::exit(-1);
    }
    return bitReversePerByte(orig, sizeof(unsigned)) >> (8 * sizeof(unsigned) - numOfBits);
}

/*  The size should be smaller than 2^32.Otherwise, the code needs modifying.
 *  The size should be power of 2.
 *  Optimization:
 *  1. reduce number of multiplications.
 *  2. reduce presicion.
 *  3. preprocess the data to make it friendly to multiplicaitons.
 *  None of above optimizations is done;
 */
std::vector<std::complex<double>> cfft(const std::vector<std::complex<double>>& samples, const bool& isInverse) {
    omp_set_num_threads(omp_get_max_threads());
    const unsigned sampleSize = unsigned(samples.size());
    auto samplesBuffer1 = std::vector<std::complex<double>>(sampleSize);
    auto samplesBuffer2 = std::vector<std::complex<double>>(sampleSize);
    const unsigned numOfBits = unsigned(std::log2(sampleSize));



    #pragma omp parallel for schedule(static, sampleSize / omp_get_max_threads() / 4)
    for (unsigned i = 0; i < sampleSize; i++) {
        samplesBuffer1[i] = samples[bitReverseInt(i, numOfBits)];
    }

    auto currentBuffer = &samplesBuffer1;
    auto nextBuffer = &samplesBuffer2;
    double inverseFactor = isInverse ? -1.0 : 1.0;

    for (unsigned i = 1; i < sampleSize; i *= 2) {
        const double numOfPointsReciprocal = 1.0 / double(i);
        #pragma omp parallel for schedule(static, sampleSize / omp_get_max_threads() / 4)
        for (unsigned j = 0; j < sampleSize; ++j) {
            (*nextBuffer)[j] = j % i < i / 2 ?
                               (*currentBuffer)[j] + oddFactor(numOfPointsReciprocal, j, inverseFactor) * (*currentBuffer)[j + i / 2] :
                               oddFactor(numOfPointsReciprocal, j, inverseFactor) * (*currentBuffer)[j] + (*currentBuffer)[j - i / 2];
        }
        std::swap(currentBuffer, nextBuffer);
    }


    if (isInverse) {
        //"parallel for" is not compatible with "for (auto x : xs)".
        const double sampleSizeReciprocal = 1.0 / double(sampleSize);
        #pragma omp parallel for schedule(static, sampleSize / omp_get_max_threads() / 4)
        for(unsigned i = 0; i < sampleSize; ++i) {
            (*nextBuffer)[i] *= sampleSizeReciprocal;
        }
    }

    return *nextBuffer;
}

std::complex<double> preBesselF(const std::complex<double>& z, const unsigned& numOfPoints, const unsigned& inputIndex) {
    using namespace std::complex_literals;
    return std::exp(1i * z * std::cos(2 * PI * double(inputIndex) / double(numOfPoints)));
}

std::vector<std::complex<double>> bessel(const std::complex<double>& z, const unsigned& order) {
    if ((order & (order - 1)) != 0) {
        std::cerr << "order must be power of 2" << std::endl;
        std::exit(-1);
    }
    auto xs = std::vector<std::complex<double>>(order);
    for (unsigned i = 0; i < xs.capacity(); ++i) {
        xs[i] = preBesselF(z, order, i);
    }
    auto begin = std::chrono::steady_clock::now();
    auto Xs = cfft(xs, true);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "mine : " << z << " : " << timeCost.count() << "s." << std::endl;
    using namespace std::complex_literals;
    for (unsigned l = 0; l < order; ++l) {
        Xs[l] *= std::pow(1i, -l) * 2.0;
    }
    return Xs;
}

std::vector<std::complex<double>> besselW(const std::complex<double>& z, const unsigned& order) {
    if ((order & (order - 1)) != 0) {
        std::cerr << "order must be power of 2" << std::endl;
        std::exit(-1);
    }
    if (fftw_init_threads() == 0) {
        std::cerr << "fftw failed to init multithreads" << std::endl;
        std::exit(-1);
    }
    fftw_complex *in, *out;
    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * order);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * order);
    for (unsigned i = 0; i < order; ++i) {
        auto xs = preBesselF(z, order, i);
        in[i][0] = xs.real();
        in[i][1] = xs.imag();
    }
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_plan p = fftw_plan_dft_1d(int(order), in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    auto begin = std::chrono::steady_clock::now();
    fftw_execute(p);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "fftw : " << z << " : " << timeCost.count() << "s." << std::endl;
    auto Xs = std::vector<std::complex<double>>(order);
    using namespace std::complex_literals;
    for (unsigned l = 0; l < order; ++l) {
        Xs[l] = (out[l][0] + out[l][1] * 1i) * std::pow(1i, -l) * 2.0;
    }
    fftw_destroy_plan(p);
    fftw_cleanup_threads();
    fftw_free(in);
    fftw_free(out);
    return Xs;
}

//results are more accurate than homework1.
int main()
{
    namespace plt = matplotlibcpp;
    const unsigned ORDER = unsigned(std::pow(2, 24));
    std::vector<double> xs(ORDER);
    std::iota(xs.begin(), xs.end(), 0);
//    for (unsigned i = 0; i < 3; ++i) {
//        auto ys = bessel(std::complex<double>(std::pow(10, i), 0), ORDER);
//        auto ysr = std::vector<double>(ORDER);
//        for (unsigned j = 0; j < ORDER; ++j) {
//            ysr[j] = ys[j].real();
//        }
//        plt::named_plot(std::to_string(std::pow(10, i)), xs, ysr);
//        std::cout << "First ten terms of J(" << std::pow(10, i) << ", n)" << std::endl;
//        std::cout << "n : J" << std::endl;
//        for (unsigned j = 0; j < 10; ++j) {
//            std::cout << j << " : " << ys[j] << std::endl;
//        }
//    }std::complex<double>(std::pow(10, i), 0), ORDER
//    plt::legend();
//    plt::xlim(0, 32);
//    plt::show();
    for (unsigned i = 0; i < 3; ++i) {
        auto fftwys = besselW(std::complex<double>(std::pow(10, i), 0), ORDER);
        auto myys = bessel(std::complex<double>(std::pow(10, i), 0), ORDER);
    }
    return 0;
}
