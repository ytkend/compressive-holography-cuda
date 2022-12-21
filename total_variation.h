#pragma once

#include <vector>
#include <complex>
#include <utility>
#include <cassert>
#include <omp.h>

float total_variation(const float* u, int nx, int ny);
float total_variation(const std::complex<float>* u, int nx, int ny);
void denoise_tv(const float* in, float* out, int nx, int ny, float weight, int nit);
void denoise_tv(const std::complex<float>* in, std::complex<float>* out, int nx, int ny, float weight, int nit);
