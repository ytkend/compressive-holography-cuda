#include <iostream>
#include <string>
#include <cassert>
#include "common.h"
#include "image.h"
#include "common.cuh"
#include "compressive_holography.cuh"

int main(int argc, char* argv[])
{
    // optimization parameters
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
        nit = std::stoi(argv[2]);       // 800 or 400
        weight = std::stof(argv[3]);    // 0.001 or 0.004
        std::cout << reg_type << ", iterations = " << nit << ", weight = " << weight << std::endl;
    }

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
            holo[x + y * nx] = {img.at(x, y) / 255.0, 0.0};
        }
    }

    // reconstruction distances (z[0] = 0)
    std::vector<float> zs(nz, 0);
    for (int i = 1; i < nz; ++i) {
        zs[i] = (i - 1) * dz + z_offset;
    }

    // check reconstruction by adjoint
    {
        // send hologram from host to device
        gpu::device_vector<gpu::complex64> d_holo(holo.size());
        thrust::copy(holo.begin(), holo.end(), d_holo.begin());

        // adjoint reconstruction
        gpu::device_vector<gpu::complex64> d_recon(holo.size() * nz);
        gpu::CSHoloModel model(nx, ny, dx, dy, wl, zs, nx, ny);
        model.adjoint(d_holo, d_recon);

        // send reconstructed images from device to host
        std::vector<complex64> recon(d_recon.size());
        thrust::copy(d_recon.begin(), d_recon.end(), recon.begin());

        // save reconstructed images
        auto imgs = abs(recon);
        for (int z = 0; z < nz; ++z) {
            auto file = "gpu_adjoint_" + std::to_string(z) + ".png";
            std::cout << "Save: " << file << std::endl;
            imsave(file, imgs.data() + holo.size() * z, nx, ny);
        }
    }

    // start timer
    CPUTimer timer;
    timer.start();

    // send hologram from host to device
    gpu::device_vector<gpu::complex64> d_holo(holo.size());
    thrust::copy(holo.begin(), holo.end(), d_holo.begin());

    // remove DC component
    auto d_holo2 = gpu::remove_dc(d_holo);

    // forward model
    gpu::CSHoloModel model(nx, ny, dx, dy, wl, zs, nx, ny);

    // fista
    auto stepsize = 0.5 / nz;
    gpu::device_vector<gpu::complex64> d_recon(nx * ny * nz);

    if (reg_type == "l1") {
        gpu::L1Norm reg;
        gpu::fista(d_holo2, d_recon, model, reg, nit, stepsize, weight, verbose);
    } else if (reg_type == "tv") {
        gpu::TV23 reg(nx, ny, nz, nit_tv);
        gpu::fista(d_holo2, d_recon, model, reg, nit, stepsize, weight, verbose);
    }

    // send hologram from device to host
    std::vector<complex64> recon(d_recon.size());
    thrust::copy(d_recon.begin(), d_recon.end(), recon.begin());

    // wait kernel execution and stop timer
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "Elapsed time: " << timer.elapsed() << " [ms]" << std::endl;

    // save reconstructed images
    auto imgs = abs(recon);
    for (int z = 0; z < nz; ++z) {
        auto file = "gpu_fista_" + std::to_string(z) + ".png";
        std::cout << "Save: " << file << std::endl;
        imsave(file, imgs.data() + holo.size() * z, nx, ny);
    }

    return 0;
}
