#include "compressive_holography.h"

// exp{ j pi * [a2 * (x^2 + y^2) + a1 * (x + y) + a0] }
void quad_phase(complex64* phase, int nx, int ny, float dx, float dy, float a0, float a1, float a2, bool band_limit)
{
    // Nyquist frequency
    float fnx = 0.5 / dx;
    float fny = 0.5 / dy;

    #pragma omp parallel for
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int i = ix + iy * nx;
            float x = dx * (ix - (nx + 1) / 2);
            float y = dy * (iy - (ny + 1) / 2);
            float arg = (float)M_PI * (a0 + a1 * (x + y) + a2 * (x * x + y * y));
            phase[i] = std::polar(1.0f, arg);

            // band-limit by local frequency
            if (band_limit) {
                float flx = std::abs(0.5f * a1 + a2 * x);
                float fly = std::abs(0.5f * a1 + a2 * y);
                if ((flx > fnx) || (fly > fny)) {
                    phase[i] = 0;
                }
            }
        }
    }
}

void fresnel_tf(complex64* tf, int nx, int ny, float dx, float dy, float wl, float z, bool band_limit = true)
{
    auto dfx = 1.0 / (nx * dx);
    auto dfy = 1.0 / (ny * dy);
    auto a0 = 2.0 / wl * z;
    auto a2 = -wl * z;
    quad_phase(tf, nx, ny, dfx, dfy, a0, 0, a2, band_limit);
    fftshift(tf, tf, ny, nx);
}

CompressiveHolographyModel::CompressiveHolographyModel(int nx, int ny, float dx, float dy, float wl, std::vector<float> zs, int pad_x, int pad_y)
    : nx(nx), ny(ny), nz(zs.size()), nx2(nx + pad_x), ny2(ny + pad_y),
      fft(ny2, nx2), fft_batch(ny2, nx2, nz), tf(nx2 * ny2 * nz), buffer(nx2 * ny2 * nz)
{
    for (int i = 0; i < nz; ++i) {
        fresnel_tf(tf.data() + i * nx2 * ny2, nx2, ny2, dx, dy, wl, zs[i]);
    }
}

void CompressiveHolographyModel::forward(const std::vector<complex64>& in, std::vector<complex64>& out)
{
    assert(in.size() >= nx * ny * nz);
    assert(out.size() >= nx * ny);

    if (nx == nx2 && ny == ny2) {
        fft_batch.forward(in.data(), buffer.data());
    } else {
        for (int z = 0; z < nz; ++z) {
            pad(in.data() + nx * ny * z,
                buffer.data() + nx2 * ny2 * z,
                ny, nx, 0, 0, ny2 - ny, nx2 - nx);
        }
        fft_batch.forward(buffer.data(), buffer.data());
    }

    #pragma omp parallel for
    for (int i = 0; i < nx2 * ny2; ++i) {
        complex64 tmp(0, 0);
        for (int z = 0; z < nz; ++z) {
            tmp += buffer[i + z * nx2 * ny2] * tf[i + z * nx2 * ny2];
        }
        buffer[i] = tmp;
    }

    if (nx == nx2 && ny == ny2) {
        fft.inverse(buffer.data(), out.data());
    } else {
        fft.inverse(buffer.data(), buffer.data());
        crop(buffer.data(), out.data(), nx2, 0, 0, ny, nx);
    }

    std::transform(out.begin(), out.begin() + nx * ny, out.begin(),
                   [](complex64 z){ return complex64(z.real(), 0); });
}

void CompressiveHolographyModel::adjoint(const std::vector<complex64>& in, std::vector<complex64>& out)
{
    assert(in.size() >= nx * ny);
    assert(out.size() >= nx * ny * nz);

    std::vector<complex64> in2(nx2 * ny2);
    if (nx == nx2 && ny == ny2) {
        fft.forward(in.data(), in2.data());
    } else {
        pad(in.data(), in2.data(), ny, nx, 0, 0, ny2 - ny, nx2 - nx);
        fft.forward(in2.data(), in2.data());
    }

    #pragma omp parallel for
    for (int z = 0; z < nz; ++z) {
        for (int i = 0; i < nx2 * ny2; ++i) {
            buffer[i + z * nx2 * ny2] = in2[i] * conj(tf[i + z * nx2 * ny2]);
        }
    }

    if (nx == nx2 && ny == ny2) {
        fft_batch.inverse(buffer.data(), out.data());
    } else {
        fft_batch.inverse(buffer.data(), buffer.data());
        for (int z = 0; z < nz; ++z) {
            crop(buffer.data() + nx2 * ny2 * z,
                 out.data() + nx * ny * z,
                 nx2, 0, 0, ny, nx);
        }
    }
}
