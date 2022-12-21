#pragma once

#include <vector>
#include <cassert>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include "common.cuh"
#include "fft.cuh"
#include "fista.cuh"
#include "total_variation.cuh"

namespace gpu {

struct CSHoloModel : ForwardModel<complex64>
{
    int nx, ny, nz;
    int nx2, ny2;
    FFT fft;
    FFTMany fft_batch;
    device_vector<complex64> tf;
    device_vector<complex64> buffer;
    dim3 block, grid;

    CSHoloModel(int nx, int ny, float dx, float dy, float wl, std::vector<float> zs, int pad_x = 0, int pad_y = 0);
    void forward(const device_vector<complex64>& in, device_vector<complex64>& out) override;
    void adjoint(const device_vector<complex64>& in, device_vector<complex64>& out) override;
};

inline device_vector<complex64> remove_dc(const device_vector<complex64>& in)
{
    float mean = thrust::transform_reduce(in.begin(), in.end(),
                                          [] __device__ (complex64 z) { return thrust::abs(z); },
                                          0.0, thrust::plus<float>());
    mean /= in.size();

    device_vector<complex64> out(in.size());
    thrust::transform(in.begin(), in.end(), out.begin(),
                      [mean] __device__ (complex64 z) { return z - mean; });
    return out;
}

__device__ inline complex64 soft(complex64 x, float t)
{
    float x_abs = abs(x);
    float x_angle = arg(x);
    x_abs = (x_abs > t) ? x_abs - t : 0;
    return thrust::polar(x_abs, x_angle);
}

struct L1Norm : ProxLoss<complex64>
{
    float operator()(const device_vector<complex64>& x) override
    {
        return thrust::transform_reduce(x.begin(), x.end(),
                                        [] __device__ (complex64 z) { return thrust::abs(z); },
                                        0.0, thrust::plus<float>());
    }

    void prox(const device_vector<complex64>& x, device_vector<complex64>& y, float t) override
    {
        thrust::transform(x.begin(), x.end(), y.begin(),
                         [t] __device__ (complex64 z) { return soft(z, t); });
    }
};

struct TV23 : ProxLoss<complex64>
{
    int nx, ny, nz;
    int nit;

    TV23(int nx, int ny, int nz, int nit) : nx(nx), ny(ny), nz(nz), nit(nit) {}

    float operator()(const device_vector<complex64>& x) override
    {
        assert(x.size() >= nx * ny * nz);

        float cost = 0;
        for (int z = 0; z < nz; ++z) {
            cost += total_variation(to_ptr(x) + z * nx * ny, nx, ny);
        }
        return cost;
    }

    void prox(const device_vector<complex64>& x, device_vector<complex64>& y, float t) override
    {
        assert(x.size() >= nx * ny * nz);
        assert(y.size() >= nx * ny * nz);

        for (int z = 0; z < nz; ++z) {
            denoise_tv(to_ptr(x) + z * nx * ny, to_ptr(y) + z * nx * ny, nx, ny, t, nit);
        }
    }
};

} // namespace gpu