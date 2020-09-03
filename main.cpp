#include<iostream>
#include<vector>
#include<tuple>
#include<complex>
#include<cmath>
#include<memory>
#include<bitset>

const double PI = std::acos(-1.0);

std::complex<double> oddFactor(const unsigned numOfPoints, const unsigned inputIndex) {
    using namespace std::complex_literals;
    return std::exp(-2.0 * PI * double(inputIndex) * 1i / double(numOfPoints));
}

std::complex<double> intermediateNode(const std::vector<std::complex<double>> &bitReversedSamples, unsigned numOfPoints, unsigned inputIndex) {
    if (numOfPoints == 1) {
        return bitReversedSamples[inputIndex];
    }
    auto preNumOfPoints = numOfPoints / 2;
    return intermediateNode(bitReversedSamples, preNumOfPoints, inputIndex) + oddFactor(numOfPoints, inputIndex) * intermediateNode(bitReversedSamples, preNumOfPoints, inputIndex < preNumOfPoints ? inputIndex + preNumOfPoints : inputIndex - preNumOfPoints);
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

unsigned bitReverseInt(unsigned orig, unsigned numOfBits) {
    // numOfBits <= 32
    return bitReverse32(orig) >> (32 - numOfBits);
}

std::vector<std::complex<double>> cfft(std::vector<std::complex<double>> samples, bool isInverse) {
    // TO DO: the bit reversal
    //from one-point to the whole
    std::vector<std::complex<double>> samplesBuffer(samples.size());
    const unsigned numOfBits = unsigned(std::log2(samples.size()));
    for (unsigned i = 0; i < samples.size(); i++) {
        samplesBuffer[i] = samples[bitReverseInt(i, numOfBits)];
    }
    auto currentBuffer = std::unique_ptr<std::vector<std::complex<double>>>(&samplesBuffer);
    auto nextBuffer = std::unique_ptr<std::vector<std::complex<double>>>(&samples);
    for (unsigned i = 2; i <= samples.size(); i *= 2) {
        for (unsigned j = 0; j < samples.size(); ++j) {
            (*nextBuffer)[j] = j % i < i / 2 ? (*currentBuffer)[j] + oddFactor(i, j) * (*currentBuffer)[j + i / 2] : oddFactor(i, j) * (*currentBuffer)[j] + (*currentBuffer)[j - i / 2];
        }
        if (currentBuffer.get() == &samples) {
            currentBuffer.release();
            nextBuffer.release();
            currentBuffer.reset(&samplesBuffer);
            nextBuffer.reset(&samples);
        } else {
            currentBuffer.release();
            nextBuffer.release();
            currentBuffer.reset(&samples);
            nextBuffer.reset(&samplesBuffer);
        }
    }
    return *nextBuffer;
}

int main()
{

    return 0;
}
