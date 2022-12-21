#include <iostream>
#include <string>
#include <random>
#include <cassert>
#include "common.h"
#include "compressive_holography.h"

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

double speed_test(std::string reg_type, int nx, int ny, int nz, int nit)
{
    // FFTW setup
    int n_threads = omp_get_max_threads();
    std::cout << "FFTW threads: " << n_threads << std::endl;
    set_fft_threads(n_threads);

    // dummy hologram data
    auto holo = dummy_data(nx, ny);

    // reconstruction distances (z[0] = 0)
    std::vector<float> zs(nz, 0);
    for (int i = 1; i < nz; ++i) {
        zs[i] = (i - 1) * dz + z_offset;
    }

    // start timer
    CPUTimer timer;
    timer.start();

    // remove DC component
    auto holo2 = remove_dc(holo);

    // forward model without padding
    CSHoloModel model(nx, ny, dx, dy, wl, zs, 0, 0);

    // fista
    auto stepsize = 0.5 / nz;
    std::vector<complex64> recon(holo.size() * nz);

    if (reg_type == "l1") {
        L1Norm reg;
        fista(holo2, recon, model, reg, nit, stepsize, weight, false);
    } else if (reg_type == "tv") {
        int nit_tv = 10;
        TV23 reg(nx, ny, nz, nit_tv);
        fista(holo2, recon, model, reg, nit, stepsize, weight, false);
    }

    // stop timer
    timer.stop();

    // clean up FFTW multithreading resources
    cleanup_fft_threads();

    return timer.elapsed();
}

int main(int argc, char* argv[])
{
    if (argc != 6) {
        std::cout << "Usage: speed_test [l1 or tv] [nx] [ny] [nz] [nit]" << std::endl;
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

        auto time = speed_test(reg_type, nx, ny, nz, nit);
        std::cout << "time = " << time << " [ms]" << std::endl;
    }

    return 0;
}

