#include "fft.cuh"

namespace gpu {

FFT::FFT(int n) : n0_(n), n1_(1)
{
    cufftPlan1d(&plan_, n, CUFFT_C2C, 1);
}

FFT::FFT(int n0, int n1) : n0_(n0), n1_(n1)
{
    cufftPlan2d(&plan_, n0, n1, CUFFT_C2C);
}

FFT::~FFT()
{
    cufftDestroy(plan_);
}

void FFT::forward(const cufftComplex* in, cufftComplex* out)
{
    cufftExecC2C(plan_, const_cast<cufftComplex*>(in), out, CUFFT_FORWARD);
}

void FFT::forward(const thrust::complex<float>* in, thrust::complex<float>* out)
{
    forward(reinterpret_cast<const cufftComplex*>(in), reinterpret_cast<cufftComplex*>(out));
}

void FFT::inverse(const cufftComplex* in, cufftComplex* out)
{
    cufftExecC2C(plan_, const_cast<cufftComplex*>(in), out, CUFFT_INVERSE);
    float c = 1.0 / (n0_ * n1_);
    thrust::transform(thrust::device_ptr<cufftComplex>(out),
                      thrust::device_ptr<cufftComplex>(out) + n0_ * n1_,
                      thrust::device_ptr<cufftComplex>(out),
                      [c] __host__ __device__ (cufftComplex z) { return cufftComplex{c * z.x, c * z.y}; });
}

void FFT::inverse(const thrust::complex<float>* in, thrust::complex<float>* out)
{
    inverse(reinterpret_cast<const cufftComplex*>(in), reinterpret_cast<cufftComplex*>(out));
}

FFTMany::FFTMany(int n, int batch) : n0_(n), n1_(1)
{
    cufftPlan1d(&plan_, n, CUFFT_C2C, batch);
}

FFTMany::FFTMany(int n0, int n1, int batch) : n0_(n0), n1_(n1), batch_(batch)
{
    int rank = 2;
    int n[] = {n0, n1};
    int idist = n0 * n1;
    int odist = n0 * n1;
    cufftPlanMany(&plan_, rank, n,
                  NULL, 1, idist,
                  NULL, 1, odist,
                  CUFFT_C2C, batch);
}

FFTMany::~FFTMany()
{
    cufftDestroy(plan_);
}

void FFTMany::forward(const cufftComplex* in, cufftComplex* out)
{
    cufftExecC2C(plan_, const_cast<cufftComplex*>(in), out, CUFFT_FORWARD);
}

void FFTMany::forward(const thrust::complex<float>* in, thrust::complex<float>* out)
{
    forward(reinterpret_cast<const cufftComplex*>(in), reinterpret_cast<cufftComplex*>(out));
}

void FFTMany::inverse(const cufftComplex* in, cufftComplex* out)
{
    cufftExecC2C(plan_, const_cast<cufftComplex*>(in), out, CUFFT_INVERSE);
    float c = 1.0 / (n0_ * n1_);
    thrust::transform(thrust::device_ptr<cufftComplex>(out),
                      thrust::device_ptr<cufftComplex>(out) + n0_ * n1_ * batch_,
                      thrust::device_ptr<cufftComplex>(out),
                      [c] __host__ __device__ (cufftComplex z) { return cufftComplex{c * z.x, c * z.y}; });
}

void FFTMany::inverse(const thrust::complex<float>* in, thrust::complex<float>* out)
{
    inverse(reinterpret_cast<const cufftComplex*>(in), reinterpret_cast<cufftComplex*>(out));
}

}
