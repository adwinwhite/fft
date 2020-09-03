#include<iostream>
#include<random>
#include<cmath>

const double pi = 3.141592653589793;

double integrand(double x) {
    return 4.0 / (1.0 + x * x);
}

double trapezoidalIntegral(unsigned n) {
    const double stepsize = 1.0 / double(n);
    double integral = 0;
    for (unsigned i = 0; i <= n; ++i) {
        integral += integrand(double(i) * stepsize);
    }
    integral = (2.0 * integral - 6) / 2.0 / double(n);
    return integral;
}

double simpsonIntegral(unsigned n) {
    const double stepsize = 1.0 / 2.0 / double(n);
    double integral = 0;
    for (unsigned i = 0; i < n; ++i) {
        integral += integrand(2.0 * double(i) * stepsize) + 4.0 * integrand((2.0 * double(i) + 1) * stepsize) + integrand((2.0 * double(i) + 2) * stepsize);
    }
    integral /= 6.0 * double(n);
    return integral;
}

bool inUnitCircle(double x, double y) {
    return x * x + y * y < 1 ? true : false; 
}

double get_random()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<double> dis(0, 1); // rage 0 - 1
    return dis(e);
}

double montecarloIntegral(unsigned n) {
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
    std::cout << "calculated by trapezoidal rule" << std::endl;
    std::cout << "num of partitions : relative error" << std::endl;
    for (unsigned i = 1; i < 100; ++i) {
        std::cout << i << ":" << std::abs(trapezoidalIntegral(i) - pi) / pi << std::endl;
    }

    std::cout << "calculated by simpson rule" << std::endl;
    std::cout << "num of partitions : relative error" << std::endl;
    for (unsigned i = 1; i < 100; ++i) {
        std::cout << i << ":" << std::abs(simpsonIntegral(i) - pi) / pi << std::endl;
    }


    std::cout << "calculated by monte carlo method" << std::endl;
    std::cout << "num of points : relative error" << std::endl;
    for (unsigned i = 1000; i < 1100; ++i) {
        std::cout << i << ":" << std::abs(montecarloIntegral(i) - pi) / pi << std::endl;
    }
} 
