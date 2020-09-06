#include<iostream>
#include<random>
#include<cmath>
#include"matplotlibcpp.h"
#include<omp.h>

const double PI = std::acos(-1.0);

double integrand(const double x) {
    return 4.0 / (1.0 + x * x);
}

double trapezoidalIntegral(const size_t n) {
    const double stepsize = 1.0 / double(n);
    double integral = 0;
    for (size_t i = 0; i <= n; ++i) {
        integral += integrand(double(i) * stepsize);
    }
    integral = (integral - 3) / double(n);
    return integral;
}

double simpsonIntegral(const size_t n) {
    const double stepsize = 1.0 / 2.0 / double(n);
    double integral = 0;
    for (size_t i = 0; i < n; ++i) {
        integral += integrand(2.0 * double(i) * stepsize) + 4.0 * integrand((2.0 * double(i) + 1) * stepsize) + integrand((2.0 * double(i) + 2) * stepsize);
    }
    integral *= stepsize / 3;
    return integral;
}

bool inUnitCircle(const double x, const double y) {
    return x * x + y * y < 1.0 ? true : false;
}

double get_random()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<double> dis(0, 1); // rage 0 - 1
    return dis(e);
}

double montecarloIntegral(const size_t n) {
    std::default_random_engine e;
    std::uniform_real_distribution<double> dis(-1, 1); 
    size_t numOfPointsInside = 0;
    for (size_t i = 0; i < n; ++i) {
        if (inUnitCircle(dis(e), dis(e))) {
            ++numOfPointsInside;
        }
    }
    return 4.0 * double(numOfPointsInside) / double(n);
}

    

int main() {
    omp_set_num_threads(omp_get_max_threads());
    namespace plt = matplotlibcpp;
//    std::cout << "calculated by trapezoidal rule" << std::endl;
//    std::cout << "num of partitions : relative error" << std::endl;
    constexpr size_t FIGURE_SIZE = 64;
    constexpr size_t START_X = 2000;
    auto xs1 = std::vector<double>(FIGURE_SIZE);
    auto ys1 = std::vector<double>(FIGURE_SIZE);
    #pragma omp parallel for schedule(static)
    for (size_t i = START_X; i < START_X + FIGURE_SIZE; ++i) {
        double y = std::abs(trapezoidalIntegral(i) - PI) / PI;
        ys1[i - START_X] = y;
        xs1[i - START_X] = double(i);
        //std::cout << i << " : " << y << std::endl;
    }
    plt::subplot(3, 1, 1);
    plt::plot(xs1, ys1);
    plt::xlim(START_X, START_X + FIGURE_SIZE);
    plt::ylabel("trapezoidal rule");

//    std::cout << "calculated by simpson rule" << std::endl;
//    std::cout << "num of partitions : relative error" << std::endl;
    auto xs2 = std::vector<double>(FIGURE_SIZE);
    auto ys2 = std::vector<double>(FIGURE_SIZE);
    #pragma omp parallel for schedule(static)
    for (size_t i = START_X; i < START_X + FIGURE_SIZE; ++i) {
        double y = std::abs(simpsonIntegral(i) - PI) / PI;
        ys2[i - START_X] = y;
        xs2[i - START_X] = double(i);
        //std::cout << i << " : " << y << std::endl;
    }
    plt::subplot(3, 1, 2);
    plt::plot(xs2, ys2);
    plt::xlim(START_X, START_X + FIGURE_SIZE);
    plt::ylabel("simpson rule");

//    std::cout << "calculated by monte carlo method" << std::endl;
//    std::cout << "num of points : relative error" << std::endl;
    auto xs3 = std::vector<double>(FIGURE_SIZE);
    auto ys3 = std::vector<double>(FIGURE_SIZE);
    #pragma omp parallel for schedule(static)
    for (size_t i = START_X; i < START_X + FIGURE_SIZE; ++i) {
        double y = std::abs(montecarloIntegral(i) - PI) / PI;
        ys3[i - START_X] = y;
        xs3[i - START_X] = double(i);
        //std::cout << i << " : " << y << std::endl;
    }
    plt::subplot(3, 1, 3);
    plt::plot(xs3, ys3);
    plt::xlim(START_X, START_X + FIGURE_SIZE);
    plt::ylabel("monte carlo");
    plt::legend();
    plt::show();
    return 0;
} 
