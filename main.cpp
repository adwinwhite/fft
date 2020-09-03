#include<iostream>
#include<vector>
#include<tuple>
#include<complex>
#include<cmath>
#include<memory>
#include<bitset>
#include<chrono>

const double PI = std::acos(-1.0);

std::complex<double> oddFactor(const unsigned numOfPoints, const unsigned inputIndex, const bool isInverse) {
    using namespace std::complex_literals;
    return std::exp((isInverse ? 1.0 : -1.0) * 2i * PI * double(inputIndex) / double(numOfPoints));
}

std::complex<double> intermediateNode(const std::vector<std::complex<double>> &bitReversedSamples, const unsigned numOfPoints, const unsigned inputIndex, const bool isInverse) {
    if (numOfPoints == 1) {
        return bitReversedSamples[inputIndex];
    }
    auto preNumOfPoints = numOfPoints / 2;
    return intermediateNode(bitReversedSamples, preNumOfPoints, inputIndex, isInverse) + oddFactor(numOfPoints, inputIndex, isInverse) * intermediateNode(bitReversedSamples, preNumOfPoints, inputIndex < preNumOfPoints ? inputIndex + preNumOfPoints : inputIndex - preNumOfPoints, isInverse);
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
        throw "The size of bits to be reversed should be smaller than 32";
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
        #pragma omp for schedule(dynamic, 5)
        for (unsigned i = 0; i < sampleSize; i++) {
            samplesBuffer[i] = samples[bitReverseInt(i, numOfBits)];
        }
    }

    auto currentBuffer = std::unique_ptr<std::vector<std::complex<double>>>(&samplesBuffer);
    auto nextBuffer = std::unique_ptr<std::vector<std::complex<double>>>(&samples);


    for (unsigned i = 2; i <= sampleSize; i *= 2) {
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 5)
            for (unsigned j = 0; j < sampleSize; ++j) {
                (*nextBuffer)[j] = j % i < i / 2 ? (*currentBuffer)[j] + oddFactor(i, j, isInverse) * (*currentBuffer)[j + i / 2] : oddFactor(i, j, isInverse) * (*currentBuffer)[j] + (*currentBuffer)[j - i / 2];
            }
        }
        currentBuffer.swap(nextBuffer);
    }


    if (isInverse) {
        #pragma omp parallel for
        for(auto &c : *nextBuffer) {
            c /= double(sampleSize);
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

int main()
{
    double zr, zi;
    std::cout << "Please enter the variable z of Bessel function:" << std::endl;
    std::cin >> zr >> zi;
    auto z = std::complex<double>(zr, zi);

    unsigned numOfIntervals;
    std::cout << "Please enter the size of samples n (actual sample's size would be nth power of 2)" << std::endl;
    std::cin >> numOfIntervals;
    numOfIntervals = unsigned(std::pow(2, numOfIntervals));

    auto xs = std::vector<std::complex<double>>(numOfIntervals);
    for (unsigned i = 0; i< xs.capacity(); ++i) {
        xs[i] = preBesselF(z, numOfIntervals, i);
        std::cout << xs[i] << std::endl;
    }

    auto startTime = std::chrono::steady_clock::now();
    auto Xs = cfft(xs, true);
    auto endTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeCost = endTime - startTime;
    std::cout << "fft costs " << timeCost.count() << "s" << std::endl;
    using namespace std::complex_literals;
    for (unsigned l = 0; l < numOfIntervals; ++l) {
        std::cout << l << ":" << Xs[l] * std::pow(1i, -l) << std::endl;
    }
    return 0;
}
