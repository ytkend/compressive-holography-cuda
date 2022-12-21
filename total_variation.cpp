#include "total_variation.h"

template <class T>
inline T norm2(T x1, T x2)
{
    return std::sqrt(x1 * x1 + x2 * x2);
}

template <class T>
inline T norm2(std::complex<T> z1, std::complex<T> z2)
{
    auto a1 = z1.real() * z1.real() + z1.imag() * z1.imag();
    auto a2 = z2.real() * z2.real() + z2.imag() * z2.imag();
    return std::sqrt(a1 + a2);
}

template <class T>
void grad_tv(const T* u, T* gx, T* gy, int nx, int ny)
{
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int i = x + y * nx;
            gx[i] = (x < nx - 1) ? u[i] - u[(x + 1) + y * nx] : 0;
            gy[i] = (y < ny - 1) ? u[i] - u[x + (y + 1) * nx] : 0;
        }
    }
}

float total_variation(const float* u, int nx, int ny)
{
    std::vector<float> gx(nx * ny);
    std::vector<float> gy(nx * ny);
    grad_tv(u, gx.data(), gy.data(), nx, ny);
    float val = 0;
    for (int i = 0; i < nx * ny; ++i) {
        val += norm2(gx[i], gy[i]);
    }
    return val;
}

float total_variation(const std::complex<float>* u, int nx, int ny)
{
    std::vector<std::complex<float>> gx(nx * ny);
    std::vector<std::complex<float>> gy(nx * ny);
    grad_tv(u, gx.data(), gy.data(), nx, ny);
    float val = 0;
    for (int i = 0; i < nx * ny; ++i) {
        val += norm2(gx[i], gy[i]);
    }
    return val;
}

void denoise_tv_(const float* in, float* out, int nx, int ny, float weight, int nit)
{
    assert(in != out);

    float stepsize = 1.0 / (8.0 * weight);

    std::vector<float> gx(nx * ny, 0);
    std::vector<float> gy(nx * ny, 0);

    std::copy(in, in + nx * ny, out);

    #pragma omp parallel
    for (int it = 0; it < nit; ++it) {
        #pragma omp for
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int i = x + y * nx;
                auto gx_o = (x < nx - 1) ? out[i] - out[(x + 1) + y * nx] : 0;
                auto gy_o = (y < ny - 1) ? out[i] - out[x + (y + 1) * nx] : 0;

                gx_o = gx[i] + stepsize * gx_o;
                gy_o = gy[i] + stepsize * gy_o;
                auto a = norm2(gx_o, gy_o);
                gx[i] = (a < 1) ? gx_o : gx_o / a;
                gy[i] = (a < 1) ? gy_o : gy_o / a;
            }
        }

        #pragma omp barrier

        #pragma omp for
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int i = x + y * nx;
                float div = 0;
                div += (x != 0) ? gx[i] - gx[(x - 1) + y * nx] : gx[i];
                div += (y != 0) ? gy[i] - gy[x + (y - 1) * nx] : gy[i];
                out[i] = in[i] - weight * div;
            }
        }
    }
}

void denoise_tv_(const std::complex<float>* in, std::complex<float>* out, int nx, int ny, float weight, int nit)
{
    assert(in != out);

    float stepsize = 1.0 / (8.0 * weight);
    std::copy(in, in + nx * ny, out);

    std::vector<std::complex<float>> gx(nx * ny, 0);
    std::vector<std::complex<float>> gy(nx * ny, 0);

    #pragma omp parallel
    for (int it = 0; it < nit; ++it) {
        #pragma omp for
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int i = x + y * nx;
                auto gx_o = (x < nx - 1) ? out[i] - out[(x + 1) + y * nx] : 0;
                auto gy_o = (y < ny - 1) ? out[i] - out[x + (y + 1) * nx] : 0;

                gx_o = gx[i] + stepsize * gx_o;
                gy_o = gy[i] + stepsize * gy_o;
                auto a = norm2(gx_o, gy_o);
                gx[i] = (a < 1) ? gx_o : gx_o / a;
                gy[i] = (a < 1) ? gy_o : gy_o / a;
            }
        }

        #pragma omp barrier

        #pragma omp for
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int i = x + y * nx;
                std::complex<float> div = 0;
                div += (x != 0) ? gx[i] - gx[(x - 1) + y * nx] : gx[i];
                div += (y != 0) ? gy[i] - gy[x + (y - 1) * nx] : gy[i];
                out[i] = in[i] - weight * div;
            }
        }
    }
}

void denoise_tv(const float* in, float* out, int nx, int ny, float weight, int nit)
{
    if (in == out) {
        std::vector<float> in2(in, in + nx * ny);
        denoise_tv_(in2.data(), out, nx, ny, weight, nit);
    } else {
        denoise_tv_(in, out, nx, ny, weight, nit);
    }
}

void denoise_tv(const std::complex<float>* in, std::complex<float>* out, int nx, int ny, float weight, int nit)
{
    if (in == out) {
        std::vector<std::complex<float>> in2(in, in + nx * ny);
        denoise_tv_(in2.data(), out, nx, ny, weight, nit);
    } else {
        denoise_tv_(in, out, nx, ny, weight, nit);
    }
}
