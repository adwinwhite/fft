#include<iostream>
#include<vector>
#include<cmath>
#include<numeric>
#include<string>
#include<algorithm>
#include"matplotlibcpp.h"

const double EPSILON = std::pow(2, -52);
const double FACTOR = std::pow(2, 52);

std::vector<double> besselM(const double x, const unsigned m) {
    auto ys = std::vector<double>(m + 1);
    ys[m] = 0;
    ys[m - 1] = 1;
    const double yrcp = 2.0 / x;
    for (unsigned i = m - 1; i > 0; --i){
        //results differ from the answers by a factor which is kind of inversely proportional to x; the sum identity implementation may have some problem.
        double y = double(i) * ys[i] * yrcp - ys[i + 1];
        //if y is smaller than the precision of the machine.
        if (y < EPSILON) {
            //std::cout << "reached the smallest precision" << std::endl;
            y *= FACTOR;
            for (unsigned j = i; j < m; ++j) {
                ys[j] *= FACTOR;
            }
        } else if (y > FACTOR) {
            //std::cout << "reached the upper bound" << std::endl;
            y *= EPSILON;
            for (unsigned j = i; j < m; ++j) {
                ys[j] *= EPSILON;
            }
        }
        ys[i - 1] = y;
    }

    // Normalize the values by the sum identity.
    double sum = 0;
    for (auto y : ys) {
        sum += y*y;
    }
    sum = sum * 2.0 - ys[0] * ys[0]; // forgot to square ys[0]. ಥ_ಥ
    const double factor = 1.0 / std::sqrt(sum);
    for (auto &y : ys) {
        y *= factor;
    }
    return ys;
}


    
int main() {
    namespace plt = matplotlibcpp;
    constexpr unsigned ORDER = 2048;
    auto xs = std::vector<double>(ORDER + 1);
    for (unsigned i = 0; i < ORDER; ++i) {
        xs[i] = double(i);
    }
    for (unsigned i = 0; i < 3; ++i) {
        auto ys = besselM(std::pow(10, i), ORDER);
        plt::named_plot(std::to_string(std::pow(10, i)), xs, ys);
        std::cout << "First ten terms of J(" << std::pow(10, i) << ", n)" << std::endl;
        std::cout << "n : J" << std::endl;
        for (unsigned j = 0; j < 10; ++j) {
            std::cout << j << " : " << ys[j] << std::endl;
        }
    }
    plt::xlim(0, 32);
    plt::legend();
    plt::show();
    return 0;
}
