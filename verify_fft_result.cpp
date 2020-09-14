#include <iostream>
#include <cmath>
#include <fstream>
#include <omp.h>

const double PI = std::acos(-1.0);
const double TWO_PI = 2.0 * PI;

typedef double mcomplex[2];

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


mcomplex* cfftm(mcomplex* samples, const unsigned& order, const bool& isInverse) {
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


    for (unsigned i = 0; i < numOfBits; ++i) {
        const unsigned factionSize = 1 << i;
        const unsigned numOfFactions = order >> (i + 1);
        #pragma omp parallel for schedule(static)
        for (unsigned j = 0; j < numOfFactions; ++j) {
            const unsigned halfFactionSize = factionSize >> 1;
            for (unsigned k = 0; k < halfFactionSize; ++k) {
                const double tempr = phaseFcts[j * factionSize + k][0] * currentBuffer[j * factionSize + k + halfFactionSize][0] - phaseFcts[j * factionSize + k][1] * currentBuffer[j * factionSize + k + halfFactionSize][1];
                const double tempi = phaseFcts[j * factionSize + k][0] * currentBuffer[j * factionSize + k + halfFactionSize][1] + phaseFcts[j * factionSize + k][1] * currentBuffer[j * factionSize + k + halfFactionSize][0];
                nextBuffer[j * factionSize + k][0] = currentBuffer[j * factionSize + k][0] + tempr;
                nextBuffer[j * factionSize + k][1] = currentBuffer[j * factionSize + k][1] + tempi;
                nextBuffer[j * factionSize + k + halfFactionSize][0] = currentBuffer[j * factionSize + k][0] - tempr;
                nextBuffer[j * factionSize + k + halfFactionSize][1] = currentBuffer[j * factionSize + k][1] - tempi;
            }
        }
        std::swap(currentBuffer, nextBuffer);
    }
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

int main() {
    std::ifstream ifs;
    ifs.open("fft_data", std::ios::in | std::ios::binary | std::ios::ate);
    const std::streampos size = ifs.tellg();
    auto memblock = new char[size_t(size)];
    ifs.seekg(0, std::ios::beg);
    ifs.read(memblock, size);
    ifs.close();
    const auto fftInput = reinterpret_cast<mcomplex*>(memblock);
    const size_t order = size / 2 / sizeof(double);
    const auto Xs = cfftm(fftInput, order, true);
    for (size_t i = 0; i < 10; ++i) {
        std::cout << Xs[i][0] << ", " << Xs[i][1] << std::endl;
    }
    delete[] memblock;
}
