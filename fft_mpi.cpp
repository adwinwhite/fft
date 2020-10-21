#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <vector>
#include <string>
#include <mpi.h>

#include "fft_uniprocessor.h"

#define ROOT_RANK 0

const double PI = std::acos(-1.0);
const double TWO_PI = 2.0 * PI;

std::complex<double> phaseFactor(const unsigned& numOfPoints, const unsigned& inputIndex, const double& inverseFactor = -1.0) {
    using namespace std::complex_literals;
    return std::exp(inverseFactor * 2i * PI * double(inputIndex) / double(numOfPoints));
}

void writeResultToFile(const std::string& filepath, myfft::mcomplex* samples, const unsigned& numOfSamples) {
    std::ofstream ofs;
    ofs.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
    const auto data = reinterpret_cast<char*>(samples);
    ofs.write(data, numOfSamples * sizeof(double) * 2);
    ofs.close();
}

myfft::mcomplex* loadSample(const std::string& filepath, const unsigned &numOfProcessors, unsigned* numOfSamplesPer) {
    char* memblock;
    myfft::mcomplex* samples;

    std::ifstream ifs;
    ifs.open(filepath, std::ios::in | std::ios::binary | std::ios::ate);
    const std::streampos sampleSize = ifs.tellg();
    *numOfSamplesPer = unsigned(sampleSize) / sizeof(double) / 2 / unsigned(numOfProcessors);
    memblock = new char[size_t(sampleSize)];
    ifs.seekg(0, std::ios::beg);
    ifs.read(memblock, sampleSize);
    ifs.close();
    const auto originalData = reinterpret_cast<double*>(memblock);

    // Permute data to feed them to FFT of smaller size
    samples = new myfft::mcomplex[*numOfSamplesPer * numOfProcessors];
    for (unsigned i = 0; i < unsigned(numOfProcessors); ++i) {
        for (unsigned j = 0; j < *numOfSamplesPer; ++j) {
            samples[i * (*numOfSamplesPer) + j][0] = originalData[j * numOfProcessors * 2 + i];
            samples[i * (*numOfSamplesPer) + j][1] = originalData[j * numOfProcessors * 2 + i + 1];
        }
    }
    delete[] memblock;
    return samples;
}


int main(int argc, char** argv) {
    //Init mpi and obtain basic info
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //Create my complex datatype
    MPI_Datatype MPI_mcomplex;
    MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_mcomplex);
    MPI_Type_commit(&MPI_mcomplex);

    myfft::mcomplex* fftInputAll;
    myfft::mcomplex* fftInputLocal;
    myfft::mcomplex* fftOutputAll;
    unsigned numOfSamplesPer;

    // Load samples
    if (world_rank == ROOT_RANK) {
        fftInputAll = loadSample("fft_data", unsigned(world_size), &numOfSamplesPer);
    }


    MPI_Bcast(&numOfSamplesPer, 1, MPI_UNSIGNED, ROOT_RANK, MPI_COMM_WORLD);
    fftInputLocal = new myfft::mcomplex[numOfSamplesPer];
    MPI_Scatter(fftInputAll, int(numOfSamplesPer), MPI_mcomplex, fftInputLocal, int(numOfSamplesPer), MPI_mcomplex, ROOT_RANK, MPI_COMM_WORLD);

    const auto fftOutputLocal = myfft::cfft_c(fftInputLocal, numOfSamplesPer);

    // Gather data to root
    if (world_rank == ROOT_RANK) {
        fftOutputAll = new myfft::mcomplex[numOfSamplesPer * world_size];
    }
    MPI_Gather(fftOutputLocal, numOfSamplesPer, MPI_mcomplex, fftOutputAll, numOfSamplesPer, MPI_mcomplex, ROOT_RANK, MPI_COMM_WORLD);

    if (world_rank == ROOT_RANK) {
        //Combine and calculate the final results;
        if (world_size > 1) {
            myfft::mcomplex* fftOutputFinal = new myfft::mcomplex[numOfSamplesPer * world_size];
            for (unsigned i = 0; i < numOfSamplesPer; ++i) {
                for (unsigned j = 0; j < unsigned(world_size); ++j) {
                    std::complex<double> tempX = 0;
                    for (unsigned k = 0; k < unsigned(world_size); ++k) {
                        tempX += std::complex<double>{fftOutputAll[k * numOfSamplesPer + i][0], fftOutputAll[k * numOfSamplesPer + i][1]} * phaseFactor(numOfSamplesPer * world_size, (i + j * numOfSamplesPer) * k);
                    }
                    fftOutputFinal[j * numOfSamplesPer + i][0] = tempX.real();
                    fftOutputFinal[j * numOfSamplesPer + i][1] = tempX.imag();
                }
            }
            writeResultToFile("fft_transformed_data", fftOutputFinal, numOfSamplesPer * world_size);
            delete[] fftOutputFinal;
        } else {
            writeResultToFile("fft_transformed_data", fftOutputLocal, numOfSamplesPer);
        }
        delete[] fftOutputAll;
        delete[] fftInputAll;
    }
    // Free memory
    delete[] fftOutputLocal;
    delete[] fftInputLocal;



    MPI_Finalize();
}
