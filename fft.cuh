#pragma once

#include <cufft.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include "common.cuh"

namespace gpu{

class FFT
{
    cufftHandle plan_;
    int n0_, n1_;

public:
    explicit FFT(int n);
    FFT(int n0, int n1); // n1 is the fastest changing dimension
    ~FFT();

    void forward(const cufftComplex* in, cufftComplex* out);
    void forward(const thrust::complex<float>* in, thrust::complex<float>* out);
    void inverse(const cufftComplex* in, cufftComplex* out);
    void inverse(const thrust::complex<float>* in, thrust::complex<float>* out);
};

class FFTMany
{
    cufftHandle plan_;
    int n0_, n1_, batch_;

public:
    FFTMany(int n, int batch);
    FFTMany(int n0, int n1, int batch); // n1 is the fastest changing dimension
    ~FFTMany();

    void forward(const cufftComplex* in, cufftComplex* out);
    void forward(const thrust::complex<float>* in, thrust::complex<float>* out);
    void inverse(const cufftComplex* in, cufftComplex* out);
    void inverse(const thrust::complex<float>* in, thrust::complex<float>* out);
};

template <class T>
__global__ void swap_half(const T* in, T* out, int n0, int n1, int first0, int first1)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int second0 = n0 - first0;
    int second1 = n1 - first1;

    for (int i0 = tidy; i0 < n0; i0 += gridDim.y * blockDim.y) {
        for (int i1 = tidx; i1 < n1; i1 += gridDim.x * blockDim.x) {
            int index0 = (i0 < second0) ? i0 + first0 : i0 - second0;
            int index1 = (i1 < second1) ? i1 + first1 : i1 - second1;
            out[i1 + i0 * n1] = in[index1 + index0 * n1];
        }
    }
}

template <typename T>
void fftshift(const T* in, T* out, int n0, int n1)
{
    int first0 = (n0 + 1) / 2;
    int first1 = (n1 + 1) / 2;

    dim3 block = BlockSize2;
    dim3 grid = grid_size({n1, n0, 1}, block);

    if (in == out) {
        thrust::device_vector<T> in_copy(in, in + n0 * n1);
        swap_half<<<grid, block>>>(thrust::raw_pointer_cast(in_copy.data()), out, n0, n1, first0, first1);
    } else {
        swap_half<<<grid, block>>>(in, out, n0, n1, first0, first1);
    }
}

template <typename T>
void ifftshift(const T* in, T* out, int n0, int n1)
{
    int first0 = n0 / 2;
    int first1 = n1 / 2;

    dim3 block = BlockSize2;
    dim3 grid = grid_size({n1, n0, 1}, block);

    if (in == out) {
        thrust::device_vector<T> in_copy(in, in + n0 * n1);
        swap_half<<<grid, block>>>(thrust::raw_pointer_cast(in_copy.data()), out, n0, n1, first0, first1);
    } else {
        swap_half<<<grid, block>>>(in, out, n0, n1, first0, first1);
    }
}

template <class T>
__global__ void g_pad(const T* in, T* out, int2 n, int2 before, int2 after, T val)
{
    int2 n_o = {n.x + before.x + after.x, n.y + before.y + after.y};

    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;

    for (int y = tidy; y < n_o.y; y += gridDim.y * blockDim.y) {
        for (int x = tidx; x < n_o.x; x += gridDim.x * blockDim.x) {
            if (before.y <= y && y < before.y + n.y &&
                before.x <= x && x < before.x + n.x) {
                out[x + y * n_o.x] = in[(x - before.x) + (y - before.y) * n.x];
            } else {
                out[x + y * n_o.x] = val;
            }
        }
    }
}

template <class T>
void pad(const T* in, T* out, int2 n, int2 before, int2 after, T val = 0)
{
    dim3 n_o = {n.x + before.x + after.x, n.y + before.y + after.y};
    auto [grid, block] = grid_block_size2(n_o);
    g_pad<<<grid, block>>>(in, out, n, before, after, val);
}

template <class T>
device_vector<T> pad(const device_vector<T>& in, int nx, int ny, int before_x, int before_y, int after_x, int after_y, T val = 0)
{
    assert(in.size() == nx * ny);

    int nx_o = nx + before_x + after_x;
    int ny_o = ny + before_y + after_y;
    device_vector<complex64> out(nx_o * ny_o);

    auto [grid, block] = grid_block_size2({nx_o, ny_o});
    g_pad<<<grid, block>>>(to_ptr(in), to_ptr(out), {nx, ny}, {before_x, before_y}, {after_x, after_y}, val);

    return out;
}

template <class T>
__global__ void g_crop(const T* in, T* out, int2 n, int2 begin, int2 end)
{
    int2 n_o = {end.x - begin.x, end.y - begin.y};
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    for (int y = tidy; y < n_o.y; y += gridDim.y * blockDim.y) {
        for (int x = tidx; x < n_o.x; x += gridDim.x * blockDim.x) {
            out[x + y * n_o.x] = in[(x + begin.x) + (y + begin.y) * n.x];
        }
    }
}

template <class T>
void crop(const T* in, T* out, int2 n, int2 first, int2 last)
{
    dim3 n_o = {last.x - first.x, last.y - first.y};
    auto [grid, block] = grid_block_size2(n_o);
    g_crop<<<grid, block>>>(in, out, n, first, last);
}

template <class T>
void crop(const T* in, T* out, int nx, int ny, int begin_x, int begin_y, int end_x, int end_y)
{
    int nx_o = end_x - begin_x;
    int ny_o = end_y - begin_y;
    auto [grid, block] = grid_block_size2({nx_o, ny_o});
    g_crop<<<grid, block>>>(in, out, {nx, ny}, {begin_x, begin_y}, {end_x, end_y});
}

} // namespace gpu
