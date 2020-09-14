#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>

const double PI = std::acos(-1.0);

std::complex<double> preBesselF(const std::complex<double>& z, const size_t& numOfPoints, const size_t& inputIndex) {
    using namespace std::complex_literals;
    return std::exp(1i * z * std::cos(2 * PI * double(inputIndex) / double(numOfPoints)));
}

int main(int argc, char** argv) {
    if (argc < 4) {
        return -1;
    }
    double zr, zi;
    size_t order;
    std::stringstream zrs{argv[1]}, zis{argv[2]}, orders{argv[3]};
    zrs >> zr;
    zis >> zi;
    orders >> order;
    std::ofstream os;
    os.open("fft_data", std::ios::out | std::ios::binary | std::ios::trunc);
    for (size_t i = 0; i < order; ++i) {
        const auto x = preBesselF(std::complex<double>(zr, zi), order, i);
        auto xr = x.real();
        auto xi = x.imag();
        os.write(reinterpret_cast<char*>(&xr), sizeof(double));
        os.write(reinterpret_cast<char*>(&xi), sizeof(double));
    }
    os.close();
    return 0;

}
