#include<iostream>
#include<vector>
#include<cmath>
#include<numeric>
#include<string>
#include<algorithm>
#include"matplotlibcpp.h"

const double EPSILON = std::pow(2, -52);
const double FACTOR = std::pow(2, 52);


std::tuple<std::vector<double>, size_t> besselM(const double x, const size_t m) {
    auto ys = std::vector<double>(m + 1, 0);
    ys[m] = 0;
    ys[m - 1] = 1;
    const double yrcp = 2.0 / x;
    for (size_t i = m - 1; i > 0; --i){
        double y = double(i) * ys[i] * yrcp - ys[i + 1];
        //if y is a little too large.
        if (std::abs(y) > FACTOR) {
            y *= EPSILON;
            for (size_t j = i; j < m; ++j) {
                ys[j] *= EPSILON;
            }
        }
        ys[i - 1] = y;
    }

    // Normalize the values by the sum identity.
    double sum = 0;
    for (const auto &y : ys) {
        sum += y*y;
    }
    sum = sum * 2.0 - ys[0] * ys[0];
    const double factor = 1.0 / std::sqrt(sum);

    //remove elements smaller than EPSILON.
    size_t sliceEndIndex = m;
    for (size_t i = 0; i <= m; ++i) {
        ys[i] *= factor;
        if (std::abs(ys[i]) < EPSILON) {
            sliceEndIndex = i;
            break;
        }
    }
    //valarray cannot be used with matplotlib so we stick to vector.
    auto validYS = std::vector<double>(ys.begin(), ys.begin() + long(sliceEndIndex));
    return std::make_tuple(validYS, sliceEndIndex);
    //indices are exclusive.
}


    
int main() {
    namespace plt = matplotlibcpp;
    constexpr size_t ORDER = 512;
    constexpr size_t EXAMPLE_NUM = 3;
    for (size_t i = 0; i < EXAMPLE_NUM; ++i) {
        std::vector<double> ys;
        size_t endIndex;
        std::tie(ys, endIndex) = besselM(std::pow(10, i), ORDER);
        auto xs = std::vector<double>(endIndex);
        std::iota(xs.begin(), xs.end(), 0);
        plt::subplot(long(EXAMPLE_NUM), 1, long(i + 1));
        plt::plot(xs, ys);
        plt::ylabel(std::to_string(std::pow(10, i)));
        std::cout << "ten terms of J(" << std::pow(10, i) << ", n)" << std::endl;
        std::cout << "n : J" << std::endl;
        for (size_t j = 0; j < 10; ++j) {
            std::cout << j << " : " << ys[j] << std::endl;
        }
    }
    plt::show();
    return 0;
}
