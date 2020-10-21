#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

std::vector<std::complex<double>> loadSample(const std::string& filepath, const size_t& startIndex, const size_t numOfSamples) {
    char* memblock;


    std::ifstream ifs;
    ifs.open(filepath, std::ios::in | std::ios::binary | std::ios::ate);
    const std::streampos sampleSize = ifs.tellg();
    if (startIndex * sizeof (double) * 2 >= sampleSize) {
        std::cerr << "Index too large" << std::endl;
        std::exit(-1);
    } else {
        ifs.seekg(sizeof(double) * 2 * startIndex, std::ios::beg);
    }

    size_t readNum;
    if (numOfSamples == 0) {
        readNum = sampleSize / sizeof (double) / 2 - startIndex;
    } else if ((startIndex + numOfSamples) * sizeof (double) * 2 > sampleSize) {
        readNum = sampleSize / sizeof (double) / 2 - startIndex;
    } else {
        readNum = numOfSamples;
    }
    memblock = new char[readNum * sizeof(double) * 2];
    ifs.read(memblock, readNum * sizeof(double) * 2);
    ifs.close();
    const auto originalData = reinterpret_cast<double*>(memblock);

    auto samples = std::vector<std::complex<double>>(readNum);
    for (size_t i = 0; i < readNum; ++i) {
        samples[i] = std::complex<double>(originalData[i * 2], originalData[i * 2 + 1]);
    }
    delete[] memblock;
    return samples;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Not enough arguments" << std::endl;
        return -1;
    }
    size_t startIndex = 0, numOfSamples = 0;
    if (argc == 4) {
        std::stringstream nfs{argv[3]};
        nfs >> numOfSamples;
    }
    if (argc >= 3) {
        std::stringstream sfs{argv[2]};
        sfs >> startIndex;
    }
    auto samples = loadSample(std::string(argv[1]), startIndex, numOfSamples);
    for (size_t i = 0; i < samples.size(); ++i) {
        std::cout << i << " : " << samples[i] << std::endl;
    }
    return 0;
}
