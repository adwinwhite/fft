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
const double TWO_PI = 2.0 * PI;

typedef double mcomplex[2];

std::complex<double> oddFactor(const double& numOfPointsReciprocal, const unsigned& inputIndex, const double& inverseFactor = -1.0) {
    using namespace std::complex_literals;
    return std::exp(inverseFactor * 2i * PI * double(inputIndex) * numOfPointsReciprocal);
}

std::complex<double> oddFactor(const unsigned& numOfPoints, const unsigned& inputIndex, const double& inverseFactor = -1.0) {
    using namespace std::complex_literals;
    return std::exp(inverseFactor * 2i * PI * double(inputIndex) / double(numOfPoints));
}

std::vector<std::complex<double>> phaseFactors(const unsigned& numOfPoints, const double& inverseFactor = -1.0) {
    using namespace std::complex_literals;
    auto factors = std::vector<std::complex<double>>(numOfPoints);
    auto numOfPointsReciprocal = 1.0 / double(numOfPoints);
    for (unsigned j = 0; j < numOfPoints; ++j) {
        factors[j] = std::exp(inverseFactor * 2i * PI * double(j) * numOfPointsReciprocal);
    }
    return factors;
}

//slower than c++ version;
double* oddFactorM(const double& numOfPointsReciprocal, const unsigned& inputIndex, const double& inverseFactor = -1.0) {
    double* result = (double*)malloc(sizeof(mcomplex));
    result[0] = std::cos(TWO_PI * double(inputIndex) * numOfPointsReciprocal);
    result[1] = inverseFactor * std::sin(TWO_PI * double(inputIndex) * numOfPointsReciprocal);
    return result;
}

mcomplex* phaseFactorsM(const unsigned& numOfPoints, const double& inverseFactor = -1.0) {
    mcomplex* result = (mcomplex*)malloc(sizeof(mcomplex) * numOfPoints);
    const double yayFactor = TWO_PI / double(numOfPoints);
    for (unsigned j = 0; j < numOfPoints; ++j) {
        result[j][0] = std::cos(double(j) * yayFactor);
        result[j][1] = inverseFactor * std::sin(double(j) * yayFactor);
    }
    return result;
}

//do not allocate memory every time. tested: slower than previous one.
void oddFactorM(const double& numOfPointsReciprocal, const unsigned& inputIndex, mcomplex* oddFactors, const unsigned& threadIndex, const double& inverseFactor = -1.0) {
    oddFactors[threadIndex][0] = std::cos(TWO_PI * double(inputIndex) * numOfPointsReciprocal);
    oddFactors[threadIndex][1] = inverseFactor * std::sin(TWO_PI * double(inputIndex) * numOfPointsReciprocal);
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

//compatible with unsigned now
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




    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < sampleSize; ++i) {
        samplesBuffer1[i] = samples[bitReverseInt(i, numOfBits)];
    }

    auto currentBuffer = &samplesBuffer1;
    auto nextBuffer = &samplesBuffer2;
    double inverseFactor = isInverse ? -1.0 : 1.0;
    auto begin = std::chrono::steady_clock::now();
    auto phaseFcts = phaseFactors(sampleSize, inverseFactor);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "fcts_mine : " << timeCost.count() << "s." << std::endl;

//    for (unsigned i = 0; i < numOfBits; ++i) {
//        //const double numOfPointsReciprocal = 1.0 / double(i << 1);
//        const unsigned factionSize = 1 << i;
////        begin = std::chrono::steady_clock::now();
//        #pragma omp parallel for schedule(static)
//        for (unsigned j = 0; j < sampleSize; ++j) {
//            (*nextBuffer)[j] = j % (factionSize << 1) < factionSize ?
//                               (*currentBuffer)[j] + phaseFcts[(j % (factionSize << 1)) << (numOfBits - i - 1)] * (*currentBuffer)[j + factionSize] :
//                               phaseFcts[(j % (factionSize << 1)) << (numOfBits - i - 1)] * (*currentBuffer)[j] + (*currentBuffer)[j - factionSize];
//        }
//        std::swap(currentBuffer, nextBuffer);
////        end = std::chrono::steady_clock::now();
////        timeCost = end - begin;
////        std::cout << "cal_mine : " << unsigned(std::log2(i)) << " : " << timeCost.count() << "s." << std::endl;
//    }

    for (unsigned i = 1; i < sampleSize; i *= 2) {
            const double numOfPointsReciprocal = 1.0 / double(i * 2);
            #pragma omp parallel for schedule(static)
            for (unsigned j = 0; j < sampleSize; ++j) {
                (*nextBuffer)[j] = j % (i * 2) < i ?
                                   (*currentBuffer)[j] + oddFactor(numOfPointsReciprocal, j, inverseFactor) * (*currentBuffer)[j + i] :
                                   oddFactor(numOfPointsReciprocal, j, inverseFactor) * (*currentBuffer)[j] + (*currentBuffer)[j - i];
            }
            std::swap(currentBuffer, nextBuffer);
    }


    if (isInverse) {
        //"parallel for" is not compatible with "for (auto x : xs)".
        const double sampleSizeReciprocal = 1.0 / double(sampleSize);
        #pragma omp parallel for schedule(static)
        for(unsigned i = 0; i < sampleSize; ++i) {
            (*nextBuffer)[i] *= sampleSizeReciprocal;
        }
    }

    return *nextBuffer;
}


std::vector<std::complex<double>> cfftDynamic(const std::vector<std::complex<double>>& samples, const bool& isInverse) {
    omp_set_num_threads(omp_get_max_threads());
    const unsigned sampleSize = unsigned(samples.size());
    auto samplesBuffer1 = std::vector<std::complex<double>>(sampleSize);
    auto samplesBuffer2 = std::vector<std::complex<double>>(sampleSize);
    const unsigned numOfBits = unsigned(std::log2(sampleSize));



    #pragma omp parallel for schedule(dynamic, sampleSize / omp_get_max_threads() / 4)
    for (unsigned i = 0; i < sampleSize; ++i) {
        samplesBuffer1[i] = samples[bitReverseInt(i, numOfBits)];
    }

    auto currentBuffer = &samplesBuffer1;
    auto nextBuffer = &samplesBuffer2;
    double inverseFactor = isInverse ? -1.0 : 1.0;

    for (unsigned i = 1; i < sampleSize; i *= 2) {
        const double numOfPointsReciprocal = 1.0 / double(i << 1);
        #pragma omp parallel for schedule(dynamic, sampleSize / omp_get_max_threads() / 4)
        for (unsigned j = 0; j < sampleSize; ++j) {
            (*nextBuffer)[j] = j % (i << 1) < i ?
                               (*currentBuffer)[j] + oddFactor(numOfPointsReciprocal, j, inverseFactor) * (*currentBuffer)[j + i] :
                               oddFactor(numOfPointsReciprocal, j, inverseFactor) * (*currentBuffer)[j] + (*currentBuffer)[j - i];
        }
        std::swap(currentBuffer, nextBuffer);
    }


    if (isInverse) {
        //"parallel for" is not compatible with "for (auto x : xs)".
        const double sampleSizeReciprocal = 1.0 / double(sampleSize);
        #pragma omp parallel for schedule(dynamic, sampleSize / omp_get_max_threads() / 4)
        for(unsigned i = 0; i < sampleSize; ++i) {
            (*nextBuffer)[i] *= sampleSizeReciprocal;
        }
    }

    return *nextBuffer;
}

std::vector<std::complex<double>> cfftNoWait(const std::vector<std::complex<double>>& samples, const bool& isInverse) {
    auto begin = std::chrono::steady_clock::now();
    omp_set_num_threads(omp_get_max_threads());
    const unsigned sampleSize = unsigned(samples.size());
    const unsigned numOfBits = unsigned(std::log2(sampleSize));
    auto sampleBuffers = std::vector<std::vector<std::complex<double>>>(numOfBits + 1, std::vector<std::complex<double>>(sampleSize));
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "init_mine_nowait : " << timeCost.count() << "s." << std::endl;


    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < sampleSize; ++i) {
        sampleBuffers[0][i] = samples[bitReverseInt(i, numOfBits)];
    }

    double inverseFactor = isInverse ? -1.0 : 1.0;

    #pragma omp parallel for schedule(dynamic, sampleSize / omp_get_max_threads()) collapse(2)
    for (unsigned i = 1; i <= numOfBits; ++i) {
        for (unsigned j = 0; j < sampleSize; ++j) {
            sampleBuffers[i][j] = j % (1 << i) < (1 << (i - 1)) ?
                               sampleBuffers[i - 1][j] + oddFactor(unsigned(1 << i), j, inverseFactor) * sampleBuffers[i - 1][j + (1 << (i - 1))] :
                               oddFactor(unsigned(1 << i), j, inverseFactor) * sampleBuffers[i - 1][j] + sampleBuffers[i - 1][j - (1 << (i - 1))];
        }
    }


    if (isInverse) {
        //"parallel for" is not compatible with "for (auto x : xs)".
        const double sampleSizeReciprocal = 1.0 / double(sampleSize);
        #pragma omp parallel for schedule(static)
        for(unsigned i = 0; i < sampleSize; ++i) {
            sampleBuffers[numOfBits][i] *= sampleSizeReciprocal;
        }
    }

    return sampleBuffers[numOfBits];
}


//changing the complex multiplication to 3*5+ does not make it faster. Neither does reusing oddFactor memory.
mcomplex* cfftm(mcomplex* samples, const unsigned& order, const bool& isInverse) {

    omp_set_num_threads(omp_get_max_threads());
    mcomplex* sampleBuffer1 = (mcomplex*)malloc(sizeof(mcomplex) * order);
    mcomplex* sampleBuffer2 = (mcomplex*)malloc(sizeof(mcomplex) * order);
    const unsigned numOfBits = unsigned(std::log2(order));



    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < order; ++i) {
        sampleBuffer1[i][0] = samples[bitReverseInt(i, numOfBits)][0];
        sampleBuffer1[i][1] = samples[bitReverseInt(i, numOfBits)][1];
    }

    auto currentBuffer = sampleBuffer1;
    auto nextBuffer = sampleBuffer2;
    double inverseFactor = isInverse ? -1.0 : 1.0;
    auto begin = std::chrono::steady_clock::now();
    //mcomplex* oddFactors = (mcomplex*)malloc(sizeof(mcomplex) * omp_get_num_threads());
    mcomplex* phaseFcts = phaseFactorsM(order, inverseFactor);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "fcts_minc : " << timeCost.count() << "s." << std::endl;

    for (unsigned i = 0; i < numOfBits; ++i) {
        //const double numOfPointsReciprocal = 1.0 / double(i << 1);
//        begin = std::chrono::steady_clock::now();
        const unsigned factionSize = 1 << i;
        #pragma omp parallel for schedule(static)
        for (unsigned j = 0; j < order; ++j) {
//            auto oddFct = oddFactor(numOfPointsReciprocal, j, inverseFactor);

//            if (j % (i << 1) < i) {
//                nextBuffer[j][0] = currentBuffer[j][0] + oddFct.real() * currentBuffer[j + i][0] - oddFct.imag() * currentBuffer[j + i][1];
//                nextBuffer[j][1] = currentBuffer[j][1] + oddFct.real() * currentBuffer[j + i][1] + oddFct.imag() * currentBuffer[j + i][0];
//            } else {
//                nextBuffer[j][0] = currentBuffer[j - i][0] + oddFct.real() * currentBuffer[j][0] - oddFct.imag() * currentBuffer[j][1];
//                nextBuffer[j][1] = currentBuffer[j - i][1] + oddFct.real() * currentBuffer[j][1] + oddFct.imag() * currentBuffer[j][0];
//            }
//            auto tid = omp_get_thread_num();
//            oddFactorM(numOfPointsReciprocal, j, oddFactors, tid,inverseFactor);
            auto factionIndex = j % (factionSize << 1);
            auto numOfRemainedBits = numOfBits - i - 1;
            if (factionIndex < factionSize) {
                nextBuffer[j][0] = currentBuffer[j][0] + phaseFcts[factionIndex << numOfRemainedBits][0] * currentBuffer[j + factionSize][0] - phaseFcts[factionIndex << numOfRemainedBits][1] * currentBuffer[j + factionSize][1];
                nextBuffer[j][1] = currentBuffer[j][1] + phaseFcts[factionIndex << numOfRemainedBits][0] * currentBuffer[j + factionSize][1] + phaseFcts[factionIndex << numOfRemainedBits][1] * currentBuffer[j + factionSize][0];
            } else {
                nextBuffer[j][0] = currentBuffer[j - factionSize][0] + phaseFcts[factionIndex << numOfRemainedBits][0] * currentBuffer[j][0] - phaseFcts[factionIndex << numOfRemainedBits][1] * currentBuffer[j][1];
                nextBuffer[j][1] = currentBuffer[j - factionSize][1] + phaseFcts[factionIndex << numOfRemainedBits][0] * currentBuffer[j][1] + phaseFcts[factionIndex << numOfRemainedBits][1] * currentBuffer[j][0];
            }
            //free(oddFct);
        }
        std::swap(currentBuffer, nextBuffer);
//        end = std::chrono::steady_clock::now();
//        timeCost = end - begin;
//        std::cout << "cal_minc : " << unsigned(std::log2(i)) << " : " << timeCost.count() << "s." << std::endl;
    }
    //free(oddFactors);
    free(phaseFcts);


    if (isInverse) {
        //"parallel for" is not compatible with "for (auto x : xs)".
        const double sampleSizeReciprocal = 1.0 / double(order);
        #pragma omp parallel for schedule(static)
        for(unsigned i = 0; i < order; ++i) {
            nextBuffer[i][0] *= sampleSizeReciprocal;
            nextBuffer[i][1] *= sampleSizeReciprocal;
        }
    }
    free(currentBuffer);
    return nextBuffer;
}

mcomplex* cfftmt(mcomplex* samples, const unsigned& order, const bool& isInverse) {
    auto begin = std::chrono::steady_clock::now();
    omp_set_num_threads(omp_get_max_threads());
    mcomplex* sampleBuffer1 = (mcomplex*)malloc(sizeof(mcomplex) * order);
    mcomplex* sampleBuffer2 = (mcomplex*)malloc(sizeof(mcomplex) * order);
    const unsigned numOfBits = unsigned(std::log2(order));
    const double inverseFactor = isInverse ? -1.0 : 1.0;

    mcomplex* phaseFcts = (mcomplex*)malloc(sizeof(mcomplex) * order);
    const double yayFactor = TWO_PI / double(order);
    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < order; ++i) {
        sampleBuffer1[i][0] = samples[bitReverseInt(i, numOfBits)][0];
        sampleBuffer1[i][1] = samples[bitReverseInt(i, numOfBits)][1];

        phaseFcts[i][0] = std::cos(double(i) * yayFactor);
        phaseFcts[i][1] = inverseFactor * std::sin(double(i) * yayFactor);
    }

    auto currentBuffer = sampleBuffer1;
    auto nextBuffer = sampleBuffer2;

    //mcomplex* oddFactors = (mcomplex*)malloc(sizeof(mcomplex) * omp_get_num_threads());

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "init_mict : " << timeCost.count() << "s." << std::endl;

    for (unsigned i = 0; i < numOfBits; ++i) {
        //const double numOfPointsReciprocal = 1.0 / double(i << 1);
//        begin = std::chrono::steady_clock::now();
        const unsigned factionSize = 1 << i;
        #pragma omp parallel for schedule(static)
        for (unsigned j = 0; j < order; ++j) {
//            auto oddFct = oddFactor(numOfPointsReciprocal, j, inverseFactor);

//            if (j % (i << 1) < i) {
//                nextBuffer[j][0] = currentBuffer[j][0] + oddFct.real() * currentBuffer[j + i][0] - oddFct.imag() * currentBuffer[j + i][1];
//                nextBuffer[j][1] = currentBuffer[j][1] + oddFct.real() * currentBuffer[j + i][1] + oddFct.imag() * currentBuffer[j + i][0];
//            } else {
//                nextBuffer[j][0] = currentBuffer[j - i][0] + oddFct.real() * currentBuffer[j][0] - oddFct.imag() * currentBuffer[j][1];
//                nextBuffer[j][1] = currentBuffer[j - i][1] + oddFct.real() * currentBuffer[j][1] + oddFct.imag() * currentBuffer[j][0];
//            }
//            auto tid = omp_get_thread_num();
//            oddFactorM(numOfPointsReciprocal, j, oddFactors, tid,inverseFactor);
            auto factionIndex = j % (factionSize << 1);
            auto numOfRemainedBits = numOfBits - i - 1;
            if (factionIndex < factionSize) {
                nextBuffer[j][0] = currentBuffer[j][0] + phaseFcts[factionIndex << numOfRemainedBits][0] * currentBuffer[j + factionSize][0] - phaseFcts[factionIndex << numOfRemainedBits][1] * currentBuffer[j + factionSize][1];
                nextBuffer[j][1] = currentBuffer[j][1] + phaseFcts[factionIndex << numOfRemainedBits][0] * currentBuffer[j + factionSize][1] + phaseFcts[factionIndex << numOfRemainedBits][1] * currentBuffer[j + factionSize][0];
            } else {
                nextBuffer[j][0] = currentBuffer[j - factionSize][0] + phaseFcts[factionIndex << numOfRemainedBits][0] * currentBuffer[j][0] - phaseFcts[factionIndex << numOfRemainedBits][1] * currentBuffer[j][1];
                nextBuffer[j][1] = currentBuffer[j - factionSize][1] + phaseFcts[factionIndex << numOfRemainedBits][0] * currentBuffer[j][1] + phaseFcts[factionIndex << numOfRemainedBits][1] * currentBuffer[j][0];
            }
            //free(oddFct);
        }
        std::swap(currentBuffer, nextBuffer);
//        end = std::chrono::steady_clock::now();
//        timeCost = end - begin;
//        std::cout << "cal_minc : " << unsigned(std::log2(i)) << " : " << timeCost.count() << "s." << std::endl;
    }
    //free(oddFactors);
    free(phaseFcts);


    if (isInverse) {
        const double sampleSizeReciprocal = 1.0 / double(order);
        #pragma omp parallel for schedule(static)
        for(unsigned i = 0; i < order; ++i) {
            nextBuffer[i][0] *= sampleSizeReciprocal;
            nextBuffer[i][1] *= sampleSizeReciprocal;
        }
    }
    free(currentBuffer);
    return nextBuffer;
}

//unable to run.
mcomplex* cfftmNoWait(mcomplex* samples, const unsigned& order, const bool& isInverse) {
    auto begin = std::chrono::steady_clock::now();
    omp_set_num_threads(omp_get_max_threads());
    const unsigned numOfBits = unsigned(std::log2(order));
    mcomplex** sampleBuffers = (mcomplex**)malloc(sizeof(mcomplex) * order * (numOfBits + 1));
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "init_minc_nowait : " << timeCost.count() << "s." << std::endl;


    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < order; ++i) {
        sampleBuffers[0][i][0] = samples[bitReverseInt(i, numOfBits)][0];
        sampleBuffers[0][i][1] = samples[bitReverseInt(i, numOfBits)][1];
    }

    double inverseFactor = isInverse ? -1.0 : 1.0;


    #pragma omp parallel for schedule(dynamic, order / omp_get_max_threads()) collapse(2)
    for (unsigned i = 1; i <= numOfBits; ++i) {
        for (unsigned j = 0; j < order; ++j) {
            auto oddFct = oddFactorM(unsigned(1 << i), j, inverseFactor);
            if (j % (1 << i) < (1 << (i - 1))) {
                sampleBuffers[i][j][0] = sampleBuffers[i - 1][j][0] + oddFct[0] * sampleBuffers[i - 1][j + (1 << (i - 1))][0] - oddFct[1] * sampleBuffers[i - 1][j + (1 << (i - 1))][1];
                sampleBuffers[i][j][1] = sampleBuffers[i - 1][j][1] + oddFct[0] * sampleBuffers[i - 1][j + (1 << (i - 1))][1] + oddFct[1] * sampleBuffers[i - 1][j + (1 << (i - 1))][0];
            } else {
                sampleBuffers[i][j][0] = sampleBuffers[i - 1][j - (1 << (i - 1))][0] + oddFct[0] * sampleBuffers[i - 1][j][0] - oddFct[1] * sampleBuffers[i - 1][j][1];
                sampleBuffers[i][j][1] = sampleBuffers[i - 1][j - (1 << (i - 1))][1] + oddFct[0] * sampleBuffers[i - 1][j][1] + oddFct[1] * sampleBuffers[i - 1][j][0];
            }
            free(oddFct);
        }
    }


    if (isInverse) {
        //"parallel for" is not compatible with "for (auto x : xs)".
        const double sampleSizeReciprocal = 1.0 / double(order);
        #pragma omp parallel for schedule(static)
        for(unsigned i = 0; i < order; ++i) {
            sampleBuffers[numOfBits][i][0] *= sampleSizeReciprocal;
            sampleBuffers[numOfBits][i][1] *= sampleSizeReciprocal;
        }
    }
    for (unsigned i = 0; i < numOfBits; ++i) {
        free(sampleBuffers[i]);
    }
    return sampleBuffers[numOfBits];
}

std::complex<double> preBesselF(const std::complex<double>& z, const unsigned& numOfPoints, const unsigned& inputIndex) {
    using namespace std::complex_literals;
    return std::exp(1i * z * std::cos(2 * PI * double(inputIndex) / double(numOfPoints)));
}


std::vector<std::complex<double>> besselM(const std::complex<double>& z, const unsigned& order) {
    if ((order & (order - 1)) != 0) {
        std::cerr << "order must be power of 2" << std::endl;
        std::exit(-1);
    }
    mcomplex* xs = (mcomplex*)malloc(sizeof(mcomplex) * order);
    for (unsigned i = 0; i < order; ++i) {
        const auto x = preBesselF(z, order, i);
        xs[i][0] = x.real();
        xs[i][1] = x.imag();
    }
    auto begin = std::chrono::steady_clock::now();
    auto Xs = cfftm(xs, order, true);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "     minc : " << timeCost.count() << "s." << std::endl;
    auto Xsv = std::vector<std::complex<double>>(order);
    using namespace std::complex_literals;
    for (unsigned l = 0; l < order; ++l) {
        switch (l % 4) {
            case 0: Xsv[l] = Xs[l][0] * 2 + Xs[l][1] * 2i; break;
            case 1: Xsv[l] = Xs[l][1] * 2 - Xs[l][0] * 2i; break;
            case 2: Xsv[l] = Xs[l][0] * (-2) + Xs[l][1] * (-2i); break;
            case 3: Xsv[l] = Xs[l][1] * (-2) + Xs[l][0] * 2i; break;
        }
    }
    free(xs);
    free(Xs);
    return Xsv;
}

std::vector<std::complex<double>> besselMT(const std::complex<double>& z, const unsigned& order) {
    if ((order & (order - 1)) != 0) {
        std::cerr << "order must be power of 2" << std::endl;
        std::exit(-1);
    }
    mcomplex* xs = (mcomplex*)malloc(sizeof(mcomplex) * order);
    for (unsigned i = 0; i < order; ++i) {
        const auto x = preBesselF(z, order, i);
        xs[i][0] = x.real();
        xs[i][1] = x.imag();
    }
    auto begin = std::chrono::steady_clock::now();
    auto Xs = cfftmt(xs, order, true);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "     mict : " << timeCost.count() << "s." << std::endl;
    auto Xsv = std::vector<std::complex<double>>(order);
    using namespace std::complex_literals;
    for (unsigned l = 0; l < order; ++l) {
        switch (l % 4) {
            case 0: Xsv[l] = Xs[l][0] * 2 + Xs[l][1] * 2i; break;
            case 1: Xsv[l] = Xs[l][1] * 2 - Xs[l][0] * 2i; break;
            case 2: Xsv[l] = Xs[l][0] * (-2) + Xs[l][1] * (-2i); break;
            case 3: Xsv[l] = Xs[l][1] * (-2) + Xs[l][0] * 2i; break;
        }
    }
    free(xs);
    free(Xs);
    return Xsv;
}

std::vector<std::complex<double>> besselMN(const std::complex<double>& z, const unsigned& order) {
    if ((order & (order - 1)) != 0) {
        std::cerr << "order must be power of 2" << std::endl;
        std::exit(-1);
    }
    mcomplex* xs = (mcomplex*)malloc(sizeof(mcomplex) * order);
    for (unsigned i = 0; i < order; ++i) {
        const auto x = preBesselF(z, order, i);
        xs[i][0] = x.real();
        xs[i][1] = x.imag();
    }
    auto begin = std::chrono::steady_clock::now();
    auto Xs = cfftmNoWait(xs, order, true);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "mincn : " << z << " : " << timeCost.count() << "s." << std::endl;
    auto Xsv = std::vector<std::complex<double>>(order);
    using namespace std::complex_literals;
    for (unsigned l = 0; l < order; ++l) {
        switch (l % 4) {
            case 0: Xsv[l] = Xs[l][0] * 2 + Xs[l][1] * 2i; break;
            case 1: Xsv[l] = Xs[l][1] * 2 - Xs[l][0] * 2i; break;
            case 2: Xsv[l] = Xs[l][0] * (-2) + Xs[l][1] * (-2i); break;
            case 3: Xsv[l] = Xs[l][1] * (-2) + Xs[l][0] * 2i; break;
        }
    }
    free(xs);
    free(Xs);
    return Xsv;
}

std::vector<std::complex<double>> besselN(const std::complex<double>& z, const unsigned& order) {
    if ((order & (order - 1)) != 0) {
        std::cerr << "order must be power of 2" << std::endl;
        std::exit(-1);
    }
    auto xs = std::vector<std::complex<double>>(order);
    for (unsigned i = 0; i < xs.capacity(); ++i) {
        xs[i] = preBesselF(z, order, i);
    }
    auto begin = std::chrono::steady_clock::now();
    auto Xs = cfftNoWait(xs, true);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "minn : " << z << " : " << timeCost.count() << "s." << std::endl;
    using namespace std::complex_literals;
    for (unsigned l = 0; l < order; ++l) {
        Xs[l] *= std::pow(1i, -l) * 2.0;
    }
    return Xs;
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
    std::cout << "     mine : " << timeCost.count() << "s." << std::endl;
    using namespace std::complex_literals;
    for (unsigned l = 0; l < order; ++l) {
        Xs[l] *= std::pow(1i, -l) * 2.0;
    }
    return Xs;
}

std::vector<std::complex<double>> besselD(const std::complex<double>& z, const unsigned& order) {
    if ((order & (order - 1)) != 0) {
        std::cerr << "order must be power of 2" << std::endl;
        std::exit(-1);
    }
    auto xs = std::vector<std::complex<double>>(order);
    for (unsigned i = 0; i < xs.capacity(); ++i) {
        xs[i] = preBesselF(z, order, i);
    }
    auto begin = std::chrono::steady_clock::now();
    auto Xs = cfftDynamic(xs, true);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "mind : " << z << " : " << timeCost.count() << "s." << std::endl;
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
//    auto begin = std::chrono::steady_clock::now();
    fftw_plan p = fftw_plan_dft_1d(int(order), in, out, FFTW_FORWARD, FFTW_ESTIMATE);
//    auto end = std::chrono::steady_clock::now();
//    std::chrono::duration<double> timeCost = end - begin;
//    std::cout << "init_fftw : " << timeCost.count() << "s." << std::endl;
    auto begin = std::chrono::steady_clock::now();
    fftw_execute(p);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = end - begin;
    std::cout << "     fftw : " << timeCost.count() << "s." << std::endl;
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
//    std::vector<double> xs(ORDER);
//    std::iota(xs.begin(), xs.end(), 0);
//    for (unsigned i = 0; i < 3; ++i) {
//        auto ys = besselM(std::complex<double>(std::pow(10, i), 0), ORDER);
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
//    }
//    plt::legend();
//    plt::xlim(0, 32);
//    plt::show();

    for (unsigned i = 0; i < 3; ++i) {
        auto fftwys = besselW(std::complex<double>(std::pow(10, i), 0), ORDER);
        auto myys = bessel(std::complex<double>(std::pow(10, i), 0), ORDER);
        //auto myysn = besselN(std::complex<double>(std::pow(10, i), 0), ORDER);
        //auto myysd = besselD(std::complex<double>(std::pow(10, i), 0), ORDER);
        auto mycys = besselM(std::complex<double>(std::pow(10, i), 0), ORDER);
        auto myctys = besselMT(std::complex<double>(std::pow(10, i), 0), ORDER);
        //auto mycnys = besselMN(std::complex<double>(std::pow(10, i), 0), ORDER);
    }
    return 0;
}
