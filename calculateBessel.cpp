#include<iostream>
#include<vector>
#include<cmath>


std::vector<double> besselM(const double x, const unsigned m) {
    std::vector<double> ys;
    ys.push_back(0);
    ys.push_back(1);
    unsigned i = 1;
    while (i < m) {
        ys.push_back(2 * (m - i) * ys[i] / x - ys[i-1]);
        ++i;
    }

    // Normalize the values by the sum identity.
    double sum = 0;
    for (auto y : ys) {
        sum += y*y;
    }
    sum = sum * 2 - ys.back();
    double factor = std::sqrt(1 / sum);
    for (auto &y : ys) {
        y *= factor;
    }
    return ys;
}


    
int main() {
    const unsigned ORDER = 32;
    for (unsigned i = 0; i < 3; ++i) {
        auto ys = besselM(std::pow(10, i), ORDER);
        std::cout << "First ten terms of J(" << std::pow(10, i) << ", n)" << std::endl;
        std::cout << "n : J" << std::endl;
        for (unsigned j = 0; j < 10; ++j) {
            std::cout << j << " : " << ys[ORDER - j] << std::endl;
        }
    }
    return 0;
}
