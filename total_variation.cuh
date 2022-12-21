#pragma once

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/complex.h>
#include "common.cuh"

namespace gpu {

float total_variation(const float* u, int nx, int ny);
float total_variation(const complex64* u, int nx, int ny);
void denoise_tv(const float* in, float* out, int nx, int ny, float weight, int nit);
void denoise_tv(const complex64* in, complex64* out, int nx, int ny, float weight, int nit);

}
