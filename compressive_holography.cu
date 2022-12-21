#include "compressive_holography.cuh"

namespace gpu {

// exp{ j pi * [a2 * (x^2 + y^2) + a1 * (x + y) + a0] }
__global__ void quad_phase(complex64* phase, int nx, int ny, float dx, float dy, float a0, float a1, float a2, bool band_limit)
{
    // Nyquist frequency
    float fnx = 0.5f / dx;
    float fny = 0.5f / dy;

    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;

    for (int iy = tidy; iy < ny; iy += blockDim.y * gridDim.y) {
        for (int ix = tidx; ix < nx; ix += blockDim.x * gridDim.x) {
            int gid = ix + iy * nx;
            float x = dx * (ix - (nx + 1) / 2);
            float y = dy * (iy - (ny + 1) / 2);
            float arg = (float)M_PI * (a0 + a1 * (x + y) + a2 * (x * x + y * y));
            phase[gid] = thrust::polar(1.0f, arg);

            // band-limit by local frequency
            if (band_limit) {
                float flx = abs(0.5f * a1 + a2 * x);
                float fly = abs(0.5f * a1 + a2 * y);
                if ((flx > fnx) || (fly > fny)) {
                    phase[gid] = 0;
                }
            }
        }
    }
}

void fresnel_tf(complex64* tf, int nx, int ny, float dx, float dy, float wl, float z, bool band_limit = true)
{
    float dfx = 1 / (nx * dx);
    float dfy = 1 / (ny * dy);
    float a0 = 2 / wl * z;
    float a2 = -wl * z;
    auto [grid, block] = grid_block_size2({nx, ny});
    quad_phase<<<grid, block>>>(tf, nx, ny, dfx, dfy, a0, 0, a2, band_limit);
    fftshift(tf, tf, ny, nx);
}

__global__ void mults_sum(complex64* in1, complex64* in2, complex64* out, int n, int m)
{
    complex64 tmp(0, 0);
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        for (int j = 0; j < m; ++j) {
           tmp += in1[i + j * n] * in2[i + j * n];
        }
        out[i] = tmp;
    }
}

__global__ void conj_mults(complex64* in1, complex64* in2, complex64* out, int n, int m)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        for (int j = 0; j < m; ++j) {
           out[i + j * n] = in1[i] * conj(in2[i + j * n]);
        }
    }
}

CSHoloModel::CSHoloModel(int nx, int ny, float dx, float dy, float wl, std::vector<float> zs, int pad_x, int pad_y)
    : nx(nx), ny(ny), nz(zs.size()), nx2(nx + pad_x), ny2(ny + pad_y),
      fft(ny2, nx2), fft_batch(ny2, nx2, nz), tf(nx2 * ny2 * nz), buffer(nx2 * ny2 * nz),
      block(block_size1(nx2 * ny2)), grid(grid_size(nx2 * ny2, block))
{
    for (int i = 0; i < nz; ++i) {
        fresnel_tf(to_ptr(tf) + i * nx2 * ny2, nx2, ny2, dx, dy, wl, zs[i]);
    }
}

void CSHoloModel::forward(const device_vector<complex64>& in, device_vector<complex64>& out)
{
    assert(in.size() >= nx * ny * nz);
    assert(out.size() >= nx * ny);

    if (nx == nx2 && ny == ny2) {
        fft_batch.forward(to_ptr(in), to_ptr(buffer));
    } else {
        for (int z = 0; z < nz; ++z) {
            pad(to_ptr(in) + nx * ny * z,
                to_ptr(buffer) + nx2 * ny2 * z,
                {nx, ny}, {0, 0}, {nx2 - nx, ny2 - ny});
        }
        fft_batch.forward(to_ptr(buffer), to_ptr(buffer));
    }

    mults_sum<<<grid, block>>>(to_ptr(buffer), to_ptr(tf), to_ptr(buffer), nx2 * ny2, nz);

    if (nx == nx2 && ny == ny2) {
        fft.inverse(to_ptr(buffer), to_ptr(out));
    } else {
        fft.inverse(to_ptr(buffer), to_ptr(buffer));
        crop(to_ptr(buffer), to_ptr(out), {nx2, ny2}, {0, 0}, {nx, ny});
    }

    thrust::transform(out.begin(), out.begin() + nx * ny, out.begin(),
                      [] __device__ (complex64 z) { return complex64(z.real(), 0); });
}

void CSHoloModel::adjoint(const device_vector<complex64>& in, device_vector<complex64>& out)
{
    assert(in.size() >= nx * ny);
    assert(out.size() >= nx * ny * nz);

    device_vector<complex64> in2(nx2 * ny2);
    if (nx == nx2 && ny == ny2) {
        fft.forward(to_ptr(in), to_ptr(in2));
    } else {
        pad(to_ptr(in), to_ptr(in2), {nx, ny}, {0, 0}, {nx2 - nx, ny2 - ny});
        fft.forward(to_ptr(in2), to_ptr(in2));
    }

    conj_mults<<<grid, block>>>(to_ptr(in2), to_ptr(tf), to_ptr(buffer), nx2 * ny2, nz);

    if (nx == nx2 && ny == ny2) {
        fft_batch.inverse(to_ptr(buffer), to_ptr(out));
    } else {
        fft_batch.inverse(to_ptr(buffer), to_ptr(buffer));
        for (int z = 0; z < nz; ++z) {
            crop(to_ptr(buffer) + nx2 * ny2 * z,
                to_ptr(out) + nx * ny * z,
                {nx2, ny2}, {0, 0}, {nx, ny});
        }
    }
}

} // namespace gpu