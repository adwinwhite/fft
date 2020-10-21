#include "fft_uniprocessor.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <complex>

const double PI = std::acos(-1.0);
const double TWO_PI = 2.0 * PI;



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



myfft::mcomplex* myfft::cfft_c(myfft::mcomplex* samples, const unsigned& order, const bool isInverse) {
    myfft::mcomplex* sampleBuffer1 = new myfft::mcomplex[order];
    myfft::mcomplex* sampleBuffer2 = new myfft::mcomplex[order];
    const unsigned numOfBits = unsigned(std::log2(order));
    const double inverseFactor = isInverse ? -1.0 : 1.0;
    myfft::mcomplex* phaseFcts = new myfft::mcomplex[order];
    const double yayFactor = TWO_PI / double(order);


    for (unsigned i = 0; i < order; ++i) {
        sampleBuffer1[i][0] = samples[bitReverseInt(i, numOfBits)][0];
        sampleBuffer1[i][1] = samples[bitReverseInt(i, numOfBits)][1];

        phaseFcts[i][0] = std::cos(double(i) * yayFactor);
        phaseFcts[i][1] = inverseFactor * std::sin(double(i) * yayFactor);
    }

    auto currentBuffer = sampleBuffer1;
    auto nextBuffer = sampleBuffer2;


    for (unsigned i = 1; i <= numOfBits; ++i) {
        const unsigned factionSize = 1 << i;
        const unsigned numOfFactions = order >> i;
        for (unsigned j = 0; j < numOfFactions; ++j) {
            const unsigned halfFactionSize = factionSize >> 1;
            for (unsigned k = 0; k < halfFactionSize; ++k) {
                const double tempr = phaseFcts[k * numOfFactions][0] * currentBuffer[j * factionSize + k + halfFactionSize][0] - phaseFcts[k * numOfFactions][1] * currentBuffer[j * factionSize + k + halfFactionSize][1];
                const double tempi = phaseFcts[k * numOfFactions][0] * currentBuffer[j * factionSize + k + halfFactionSize][1] + phaseFcts[k * numOfFactions][1] * currentBuffer[j * factionSize + k + halfFactionSize][0];
                nextBuffer[j * factionSize + k][0] = currentBuffer[j * factionSize + k][0] + tempr;
                nextBuffer[j * factionSize + k][1] = currentBuffer[j * factionSize + k][1] + tempi;
                nextBuffer[j * factionSize + k + halfFactionSize][0] = currentBuffer[j * factionSize + k][0] - tempr;
                nextBuffer[j * factionSize + k + halfFactionSize][1] = currentBuffer[j * factionSize + k][1] - tempi;
            }
        }

        std::swap(currentBuffer, nextBuffer);
    }
    delete[] phaseFcts;


    if (isInverse) {
        const double sampleSizeReciprocal = 1.0 / double(order);
        for(unsigned i = 0; i < order; ++i) {
            currentBuffer[i][0] *= sampleSizeReciprocal;
            currentBuffer[i][1] *= sampleSizeReciprocal;
        }
    }
    delete[] nextBuffer;
    return currentBuffer;
}


myfft::mcomplex* myfft::cfft(myfft::mcomplex* samples, const unsigned& order, const bool isInverse) {
    auto samplesBuffer1 = std::vector<std::complex<double>>(order);
    auto samplesBuffer2 = std::vector<std::complex<double>>(order);
    const unsigned numOfBits = unsigned(std::log2(order));
    const double inverseFactor = isInverse ? -1.0 : 1.0;
    auto phaseFcts = std::vector<std::complex<double>>(order);
    auto orderReciprocal = 1.0 / double(order);

    auto samples_p = std::vector<std::complex<double>>(order);
    for (unsigned i = 0; i < order; ++i) {
        samples_p[i] = std::complex<double>(samples[i][0], samples[i][1]);
    }


    for (unsigned i = 0; i < order; ++i) {
        samplesBuffer1[i] = samples_p[bitReverseInt(i, numOfBits)];
        phaseFcts[i] = std::exp(inverseFactor * std::complex<double>(0, 1) * TWO_PI * double(i) * orderReciprocal);
    }

    auto currentBuffer = &samplesBuffer1;
    auto nextBuffer = &samplesBuffer2;

//    for (unsigned i = 0; i < order; ++i) {
//        for (unsigned j = 0; j < order; ++j) {
//            (*nextBuffer)[i] += samples_p[j] * std::pow(std::exp(std::complex<double>(0, 1) * TWO_PI * orderReciprocal), j * i);
//        }
//    }


    for (unsigned i = 1; i <= numOfBits; ++i) {
        const unsigned factionSize = 1 << i;
        const unsigned numOfFactions = order >> i;
        //#pragma omp parallel for schedule(static)
        for (unsigned j = 0; j < numOfFactions; ++j) {
            const unsigned halfFactionSize = factionSize >> 1;
            for (unsigned k = 0; k < halfFactionSize; ++k) {
                auto temp = phaseFcts[k * numOfFactions] * (*currentBuffer)[j * factionSize + k + halfFactionSize];
                (*nextBuffer)[j * factionSize + k] = (*currentBuffer)[j * factionSize + k] + temp;
                (*nextBuffer)[j * factionSize + k + halfFactionSize] = (*currentBuffer)[j * factionSize + k] - temp;
            }
        }
        std::swap(currentBuffer, nextBuffer);
    }




    if (isInverse) {
        const double orderReciprocal = 1.0 / double(order);
        for(unsigned i = 0; i < order; ++i) {
            (*currentBuffer)[i] *= orderReciprocal;
        }
    }

    myfft::mcomplex* sample_r = new myfft::mcomplex[order];
    for (unsigned i = 0; i < order; ++i) {
        sample_r[i][0] = (*currentBuffer)[i].real();
        sample_r[i][1] = (*currentBuffer)[i].imag();
    }
    return sample_r;
}
