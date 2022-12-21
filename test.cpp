#include <iostream>
#include <string>
#include <cassert>
#include "common.h"
#include "image.h"
#include "compressive_holography.h"

int main(int argc, char* argv[])
{
    // parameters
    bool verbose = true;
    int nit_tv = 10;
    int nit;
    float weight;
    std::string reg_type;

    if (argc != 4) {
        std::cout << "Usage: test [l1 or tv] [nit] [weight]" << std::endl;
        return 0;
    } else {
        reg_type = std::string(argv[1]);
        assert(reg_type == "l1" || reg_type == "tv");
        nit = std::stoi(argv[2]);
        weight = std::stof(argv[3]);
        std::cout << reg_type << ", iterations = " << nit << ", weight = " << weight << std::endl;
    }

    // FFTW setup
    int n_threads = omp_get_max_threads();
    std::cout << "FFTW threads: " << n_threads << std::endl;
    set_fft_threads(n_threads);

    // hologram parameters
    auto dx = 10e-6, dy = 10e-6;
    auto wl = 632.8e-9;
    auto nz = 5;
    auto z_offset = 10e-3;
    auto dz = 10e-3;

    // load hologram data
    std::string file;
    if (reg_type == "l1") {
        file = "../data/hologram_points.png";
    } else if (reg_type == "tv") {
        file = "../data/hologram_1234.png";
    }

    std::cout << "Load: " << file << std::endl;
    auto img = imread<float>(file);
    auto nx = img.width, ny = img.height;

    std::vector<complex64> holo(nx * ny);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            holo[x + y * nx] = {img.at(x, y) / 255.0f, 0.0f};
        }
    }

    // reconstruction distances (z[0] = 0)
    std::vector<float> zs(nz, 0);
    for (int i = 1; i < nz; ++i) {
        zs[i] = (i - 1) * dz + z_offset;
    }

    // check reconstruction by adjoint
    {
        CSHoloModel model(nx, ny, dx, dy, wl, zs, nx, ny);
        std::vector<complex64> recon(nx * ny * nz);
        model.adjoint(holo, recon);

        auto imgs = abs(recon);
        for (int z = 0; z < nz; ++z) {
            auto file = "cpu_adjoint_" + std::to_string(z) + ".png";
            std::cout << "Save: " << file << std::endl;
            imsave(file, imgs.data() + z * nx * ny, nx, ny);
        }
    }

    // start timer
    CPUTimer timer;
    timer.start();

    // remove DC component
    auto holo2 = remove_dc(holo);

    // forward model
    CSHoloModel model(nx, ny, dx, dy, wl, zs, nx, ny);

    // fista
    auto stepsize = 0.5 / nz;
    std::vector<complex64> recon(nx * ny * nz);

    if (reg_type == "l1") {
        L1Norm reg;
        fista(holo2, recon, model, reg, nit, stepsize, weight, verbose);
    } else if (reg_type == "tv") {
        TV23 reg(nx, ny, nz, nit_tv);
        fista(holo2, recon, model, reg, nit, stepsize, weight, verbose);
    }

    // stop timer
    timer.stop();
    std::cout << "Elapsed time: " << timer.elapsed() << " [ms]" << std::endl;

    // save reconstructed images
    auto imgs = abs(recon);
    for (int z = 0; z < nz; ++z) {
        auto file = "cpu_fista_" + std::to_string(z) + ".png";
        std::cout << "Save: " << file << std::endl;
        imsave(file, imgs.data() + z * nx * ny, nx, ny);
    }

    // clean up FFTW multithreading resources
    cleanup_fft_threads();

    return 0;
}
