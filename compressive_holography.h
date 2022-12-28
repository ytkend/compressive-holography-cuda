#pragma once

#include <vector>
#include <complex>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include "common.h"
#include "fista.h"
#include "fft.h"
#include "total_variation.h"

struct CompressiveHolographyModel : ForwardModel<complex64>
{
    int nx, ny, nz;
    int nx2, ny2;
    FFT fft;
    FFTMany fft_batch;
    std::vector<complex64> tf;
    std::vector<complex64> buffer;

    CompressiveHolographyModel(int nx, int ny, float dx, float dy, float wl, std::vector<float> zs, int pad_x = 0, int pad_y = 0);
    void forward(const std::vector<complex64>& in, std::vector<complex64>& out);
    void adjoint(const std::vector<complex64>& in, std::vector<complex64>& out);
};

inline std::vector<complex64> remove_dc(const std::vector<complex64>& in)
{
    float mean = 0;
    for (const auto& i: in) mean += std::abs(i);
    mean /= in.size();

    std::vector<complex64> out(in.size());
    for (int i = 0; i < in.size(); ++i) out[i] = in[i] - mean;
    return out;
}

inline complex64 soft(complex64 x, float t)
{
    float x_abs = std::abs(x);
    float x_angle = std::arg(x);
    x_abs = (x_abs > t) ? x_abs - t : 0;
    return std::polar(x_abs, x_angle);
}

struct L1Norm : ProxLoss<complex64>
{
    float operator()(const std::vector<complex64>& x)
    {
        auto absolute = [](complex64 z) { return std::abs(z); };
        return std::transform_reduce(x.begin(), x.end(), 0.0, std::plus<float>(), absolute);
    }

    void prox(const std::vector<complex64>& x, std::vector<complex64>& y, float t)
    {
        #pragma omp parallel for
        for (int i = 0; i < x.size(); ++i) {
            y[i] = soft(x[i], t);
        }
    }
};

struct TV23 : ProxLoss<complex64>
{
    int nx, ny, nz;
    int nit;

    TV23(int nx, int ny, int nz, int nit) : nx(nx), ny(ny), nz(nz), nit(nit) {}

    float operator()(const std::vector<complex64>& x)
    {
        assert(x.size() >= nx * ny * nz);

        float cost = 0;
        for (int z = 0; z < nz; ++z) {
            cost += total_variation(x.data() + z * nx * ny, nx, ny);
        }
        return cost;
    }

    void prox(const std::vector<complex64>& x, std::vector<complex64>& y, float t)
    {
        assert(x.size() >= nx * ny * nz);
        assert(y.size() >= nx * ny * nz);

        for (int z = 0; z < nz; ++z) {
            denoise_tv(x.data() + z * nx * ny, y.data() + z * nx * ny, nx, ny, t, nit);
        }
    }
};
