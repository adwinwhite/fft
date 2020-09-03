#include<iostream>
#include<random>
#include<cmath>
#include"matplotlibcpp.h"

const double PI = std::acos(-1.0);

double integrand(const double x) {
    return 4.0 / (1.0 + x * x);
}

double trapezoidalIntegral(const unsigned n) {
    const double stepsize = 1.0 / double(n);
    double integral = 0;
    for (unsigned i = 0; i <= n; ++i) {
        integral += integrand(double(i) * stepsize);
    }
    integral = (2.0 * integral - 6) / 2.0 / double(n);
    return integral;
}

double simpsonIntegral(const unsigned n) {
    const double stepsize = 1.0 / 2.0 / double(n);
    double integral = 0;
    for (unsigned i = 0; i < n; ++i) {
        integral += integrand(2.0 * double(i) * stepsize) + 4.0 * integrand((2.0 * double(i) + 1) * stepsize) + integrand((2.0 * double(i) + 2) * stepsize);
    }
    integral /= 6.0 * double(n);
    return integral;
}

bool inUnitCircle(const double x, const double y) {
    return x * x + y * y < 1 ? true : false; 
}

double get_random()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<double> dis(0, 1); // rage 0 - 1
    return dis(e);
}

double montecarloIntegral(const unsigned n) {
    std::default_random_engine e;
    std::uniform_real_distribution<double> dis(-1, 1); 
    unsigned numOfPointsInside = 0;
    for (unsigned i = 0; i < n; ++i) {
        if (inUnitCircle(dis(e), dis(e))) {
            ++numOfPointsInside;
        }
    }
    return 4.0 * double(numOfPointsInside) / double(n);
}

    

int main() {
    namespace plt = matplotlibcpp;
    std::cout << "calculated by trapezoidal rule" << std::endl;
    std::cout << "num of partitions : relative error" << std::endl;
    auto xs1 = std::vector<double>(20);
    auto ys1 = std::vector<double>(20);
    for (unsigned i = 80; i < 100; ++i) {
        double y = std::abs(trapezoidalIntegral(i) - PI) / PI;
        ys1.push_back(y);
        xs1.push_back(i);
        std::cout << i << " : " << y << std::endl;
    }
    plt::subplot(3, 1, 1);
    plt::plot(xs1, ys1);
    plt::xlim(80, 100);
    plt::ylabel("trapezoidal rule");

    std::cout << "calculated by simpson rule" << std::endl;
    std::cout << "num of partitions : relative error" << std::endl;
    auto xs2 = std::vector<double>(20);
    auto ys2 = std::vector<double>(20);
    for (unsigned i = 80; i < 100; ++i) {
        double y = std::abs(simpsonIntegral(i) - PI) / PI;
        ys2.push_back(y);
        xs2.push_back(i);
        std::cout << i << " : " << y << std::endl;
    }
    plt::subplot(3, 1, 2);
    plt::plot(xs2, ys2);
    plt::xlim(80, 100);
    plt::ylabel("simpson rule");

    std::cout << "calculated by monte carlo method" << std::endl;
    std::cout << "num of points : relative error" << std::endl;
    auto xs3 = std::vector<double>(20);
    auto ys3 = std::vector<double>(20);
    for (unsigned i = 1090; i < 1100; ++i) {
        double y = std::abs(montecarloIntegral(i) - PI) / PI;
        ys3.push_back(y);
        xs3.push_back(i);
        std::cout << i << " : " << y << std::endl;
    }
    plt::subplot(3, 1, 3);
    plt::plot(xs3, ys3);
    plt::xlim(1090, 1100);
    plt::ylabel("monte carlo");
    plt::legend();
    plt::show();
    return 0;
} 
