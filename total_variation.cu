#include "total_variation.cuh"

namespace gpu {

__host__ __device__ inline float norm2(float x1, float x2)
{
    return sqrtf(x1 * x1 + x2 * x2);
}

__host__ __device__ inline float norm2(complex64 z1, complex64 z2)
{
    auto a1 = z1.real() * z1.real() + z1.imag() * z1.imag();
    auto a2 = z2.real() * z2.real() + z2.imag() * z2.imag();
    return sqrtf(a1 + a2);
}

template <class T>
__global__ void grad_tv(const T* u, T* gx, T* gy, int nx, int ny)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;

    for (int y = tidy; y < ny; y += blockDim.y * gridDim.y) {
        for (int x = tidx; x < nx; x += blockDim.x * gridDim.x) {
            int i = x + y * nx;
            gx[i] = (x < nx - 1) ? u[i] - u[(x + 1) + y * nx] : 0;
            gy[i] = (y < ny - 1) ? u[i] - u[x + (y + 1) * nx] : 0;
        }
    }
}

float total_variation(const float* u, int nx, int ny)
{
    thrust::device_vector<float> gx(nx * ny);
    thrust::device_vector<float> gy(nx * ny);
    auto [grid, block] = grid_block_size2({nx, ny});
    grad_tv<<<grid, block>>>(u, thrust::raw_pointer_cast(gx.data()), thrust::raw_pointer_cast(gy.data()), nx, ny);

    thrust::device_vector<float> a(nx * ny);
    thrust::transform(gx.begin(), gx.end(), gy.begin(), a.begin(), [] __device__ (float x, float y) { return norm2(x, y); });
    return thrust::reduce(a.begin(), a.end());
}

float total_variation(const complex64* u, int nx, int ny)
{
    thrust::device_vector<complex64> gx(nx * ny);
    thrust::device_vector<complex64> gy(nx * ny);
    auto [grid, block] = grid_block_size2({nx, ny});
    grad_tv<<<grid, block>>>(u, thrust::raw_pointer_cast(gx.data()), thrust::raw_pointer_cast(gy.data()), nx, ny);

    thrust::device_vector<float> a(nx * ny);
    thrust::transform(gx.begin(), gx.end(), gy.begin(), a.begin(), [] __device__ (complex64 x, complex64 y) { return norm2(x, y); });
    return thrust::reduce(a.begin(), a.end());
}

template <class T>
__global__ void denoise_tv_dual_update(const T* u, T* gx, T* gy, int nx, int ny, T stepsize)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;

    for (int y = tidy; y < ny; y += blockDim.y * gridDim.y) {
        for (int x = tidx; x < nx; x += blockDim.x * gridDim.x) {
            int i = x + y * nx;
            auto gx_o = (x < nx - 1) ? u[i] - u[(x + 1) + y * nx] : 0;
            auto gy_o = (y < ny - 1) ? u[i] - u[x + (y + 1) * nx] : 0;

            gx_o = gx[i] + stepsize * gx_o;
            gy_o = gy[i] + stepsize * gy_o;
            auto a = norm2(gx_o, gy_o);
            gx[i] = (a < 1) ? gx_o : gx_o / a;
            gy[i] = (a < 1) ? gy_o : gy_o / a;
        }
    }
}

template <class T>
__global__ void denoise_tv_primal_update(const T* gx, const T* gy, const T* u, T* u_opt, int nx, int ny, T stepsize)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;

    for (int y = tidy; y < ny; y += blockDim.y * gridDim.y) {
        for (int x = tidx; x < nx; x += blockDim.x * gridDim.x) {
            int i = x + y * nx;
            T div = 0;
            div += (x != 0) ? gx[i] - gx[(x - 1) + y * nx] : gx[i];
            div += (y != 0) ? gy[i] - gy[x + (y - 1) * nx] : gy[i];

            u_opt[i] = u[i] - stepsize * div;
        }
    }
}

void denoise_tv_(const float* in, float* out, int nx, int ny, float weight, int nit)
{
    assert(in != out);

    float stepsize = 1.0 / (8.0 * weight);
    cudaMemcpy(out, in, sizeof(float) * nx * ny, cudaMemcpyDeviceToDevice);
    auto [grid, block] = grid_block_size2({nx, ny});

    thrust::device_vector<float> gx(nx * ny, 0);
    thrust::device_vector<float> gy(nx * ny, 0);

    for (int it = 0; it < nit; ++it) {
        denoise_tv_dual_update<<<grid, block>>>(out, to_ptr(gx), to_ptr(gy), nx, ny, stepsize);
        denoise_tv_primal_update<<<grid, block>>>(to_ptr(gx), to_ptr(gy), in, out, nx, ny, weight);
    }
}

void denoise_tv_(const complex64* in, complex64* out, int nx, int ny, float weight, int nit)
{
    assert(in != out);

    complex64 stepsize = 1.0 / (8.0 * weight);
    cudaMemcpy(out, in, sizeof(complex64) * nx * ny, cudaMemcpyDeviceToDevice);
    auto [grid, block] = grid_block_size2({nx, ny});

    thrust::device_vector<complex64> gx(nx * ny, 0);
    thrust::device_vector<complex64> gy(nx * ny, 0);

    for (int it = 0; it < nit; ++it) {
        denoise_tv_dual_update<<<grid, block>>>(out, to_ptr(gx), to_ptr(gy), nx, ny, stepsize);
        denoise_tv_primal_update<<<grid, block>>>(to_ptr(gx), to_ptr(gy), in, out, nx, ny, static_cast<complex64>(weight));
    }
}

void denoise_tv(const float* in, float* out, int nx, int ny, float weight, int nit)
{
    if (in == out) {
        thrust::device_vector<float> in2(in, in + nx * ny);
        denoise_tv_(to_ptr(in2), out, nx, ny, weight, nit);
    } else {
        denoise_tv_(in, out, nx, ny, weight, nit);
    }
}

void denoise_tv(const complex64* in, complex64* out, int nx, int ny, float weight, int nit)
{
    if (in == out) {
        thrust::device_vector<complex64> in2(in, in + nx * ny);
        denoise_tv_(to_ptr(in2), out, nx, ny, weight, nit);
    } else {
        denoise_tv_(in, out, nx, ny, weight, nit);
    }
}

} // namespace gpu