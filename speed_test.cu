#include <iostream>
#include <string>
#include <random>
#include <cassert>
#include "common.h"
#include "common.cuh"
#include "compressive_holography.cuh"

constexpr float wl = 632.8e-9;
constexpr float dx = 10e-6, dy = 10e-6;
constexpr float z_offset = 10e-3;
constexpr float dz = 10e-3;
constexpr float weight = 0.001;

std::vector<complex64> dummy_data(int nx, int ny)
{
    std::random_device rd;
    auto seed = rd();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(0, 1);
    std::vector<complex64> data(nx * ny);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            data[x + y * nx] = {dist(gen), 0.0};
        }
    }
    return data;
}

double speed_test_cu(std::string reg_type, int nx, int ny, int nz, int nit)
{
    // dummy hologram data
    auto holo = dummy_data(nx, ny);

    // reconstruction distances
    std::vector<float> zs(nz, 0);
    for (int i = 1; i < nz; ++i) {
        zs[i] = (i - 1) * dz + z_offset;
    }

    // start timer
    CPUTimer timer;
    timer.start();

    // send hologram from host to device
    gpu::device_vector<gpu::complex64> d_holo(holo.size());
    thrust::copy(holo.begin(), holo.end(), d_holo.begin());

    // remove DC component
    auto d_holo2 = gpu::remove_dc(d_holo);

    // forward model without padding
    gpu::CompressiveHolographyModel model(nx, ny, dx, dy, wl, zs, 0, 0);

    // fista
    auto stepsize = 0.5 / nz;
    gpu::device_vector<gpu::complex64> d_recon(holo.size() * nz);

    if (reg_type == "l1") {
        gpu::L1Norm reg;
        gpu::fista(d_holo2, d_recon, model, reg, nit, stepsize, weight, false);
    } else if (reg_type == "tv") {
        int nit_tv = 10;
        gpu::TV23 reg(nx, ny, nz, nit_tv);
        gpu::fista(d_holo2, d_recon, model, reg, nit, stepsize, weight, false);
    }

    // send reconstructed images from device to host
    std::vector<complex64> recon(d_recon.size());
    thrust::copy(d_recon.begin(), d_recon.end(), recon.begin());

    // wait kernel execution and stop timer
    cudaDeviceSynchronize();
    timer.stop();

    return timer.elapsed();
}

int main(int argc, char* argv[])
{
    if (argc != 6) {
        std::cout << "Usage: speed_test_cu [l1 or tv] [nx] [ny] [nz] [nit]" << std::endl;
        return 0;
    } else {
        auto reg_type = std::string(argv[1]);
        assert(reg_type == "l1" || reg_type == "tv");
        auto nx = std::stoi(argv[2]);
        auto ny = std::stoi(argv[3]);
        auto nz = std::stoi(argv[4]);
        auto nit = std::stoi(argv[5]);
        std::cout << reg_type << ": "
                  << "(nx, ny, nz) = (" << nx << ", " << ny << ", " << nz << "), "
                  << "iterations = " << nit << std::endl;

        auto time = speed_test_cu(reg_type, nx, ny, nz, nit);
        std::cout << "time = " << time << " [ms]" << std::endl;
    }

    return 0;
}
