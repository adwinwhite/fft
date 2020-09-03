#include<iostream>
#include<vector>
#include<cmath>
#include<numeric>
#include<string>
#include"matplotlibcpp.h"

const long double EPSILON = std::pow(2, -52);
const long double FACTOR = std::pow(2, 52);

std::vector<long double> besselM(const long double x, const unsigned m) {
    std::vector<long double> ys;
    ys.push_back(0);
    ys.push_back(1);
    unsigned i = 1;
    while (i < m) {
        long double y = 2 * (m - i) * ys[i] / x - ys[i-1];
        //if y is smaller than the precision of the machine.
        if (y < EPSILON) {
            y *= FACTOR;
            for (unsigned j = 0; j <= i; ++j) {
                ys[j] *= FACTOR;
            }
        }
        ys.push_back(y);
        ++i;
    }

    // Normalize the values by the sum identity.
    long double sum = 0;
    for (auto y : ys) {
        sum += y*y;
    }
    sum = sum * 2 - ys.back();
    long double factor = std::sqrt(1 / sum);
    for (auto &y : ys) {
        y *= factor;
    }
    return ys;
}


    
int main() {
    namespace plt = matplotlibcpp;
    constexpr unsigned ORDER = 512;
    auto xs = std::vector<long double>(ORDER + 1);
    for (unsigned i = 0; i < ORDER; ++i) {
        xs[i] = (long double)(i);
    }
    for (unsigned i = 0; i < 3; ++i) {
        auto ys = besselM(std::pow(10, i), ORDER);
        plt::named_plot(std::to_string(std::pow(10, i)), xs, ys);
        std::cout << "First ten terms of J(" << std::pow(10, i) << ", n)" << std::endl;
        std::cout << "n : J" << std::endl;
        for (unsigned j = 0; j < 10; ++j) {
            std::cout << j << " : " << ys[ORDER - j] << std::endl;
        }
    }
    plt::xlim(0, int(ORDER / 2));
    plt::legend();
    plt::show();
    return 0;
}
