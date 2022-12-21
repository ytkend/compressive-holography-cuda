#pragma once

#include <vector>
#include <string>
#include <complex>
#include <chrono>

struct CPUTimer
{
    using Clock = std::chrono::high_resolution_clock;

    std::chrono::time_point<Clock> t0, t1;

    CPUTimer(){}
    ~CPUTimer(){}

    void start()
    {
        t0 = Clock::now();
    }

    void stop()
    {
        t1 = Clock::now();
    }

    double elapsed(std::string format="ms") const
    {
        if (format == "ms") {
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        } else if (format == "us") {
            return std::chrono::duration<double, std::micro>(t1 - t0).count();
        } else {
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
    }
};

using complex64 = std::complex<float>;

template <class T>
std::vector<T> abs(const std::vector<std::complex<T>>& z)
{
    std::vector<T> v(z.size());
    for (int i = 0; i < z.size(); ++i) v[i] = std::abs(z[i]);
    return v;
}

template <typename T>
std::vector<T> real(class std::vector<std::complex<T>>& z)
{
    std::vector<T> v(z.size());
    for (int i = 0; i < z.size(); ++i) v[i] = z[i].real();
    return v;
}

template <typename T>
std::vector<T> imag(class std::vector<std::complex<T>>& z)
{
    std::vector<T> v(z.size());
    for (int i = 0; i < z.size(); ++i) v[i] = z[i].imag();
    return v;
}
