#include "fft.h"

FFT::FFT(int n0) : n0_(n0), n1_(1)
{
    plan_forward_ = fftwf_plan_dft_1d(n0, 0, 0, FFTW_FORWARD,  FFTW_ESTIMATE);
    plan_inverse_ = fftwf_plan_dft_1d(n0, 0, 0, FFTW_BACKWARD, FFTW_ESTIMATE);
}

FFT::FFT(int n0, int n1) : n0_(n0), n1_(n1)
{
    plan_forward_ = fftwf_plan_dft_2d(n0, n1, 0, 0, FFTW_FORWARD,  FFTW_ESTIMATE);
    plan_inverse_ = fftwf_plan_dft_2d(n0, n1, 0, 0, FFTW_BACKWARD, FFTW_ESTIMATE);

}

FFT::~FFT()
{
    fftwf_destroy_plan(plan_forward_);
    fftwf_destroy_plan(plan_inverse_);
}

void FFT::forward(const std::complex<float> *in, std::complex<float> *out) const
{
    fftwf_execute_dft(plan_forward_,
                      reinterpret_cast<fftwf_complex*>(const_cast<std::complex<float>*>(in)),
                      reinterpret_cast<fftwf_complex*>(out));
}

void FFT::inverse(const std::complex<float> *in, std::complex<float> *out) const
{
    fftwf_execute_dft(plan_inverse_,
                      reinterpret_cast<fftwf_complex*>(const_cast<std::complex<float>*>(in)),
                      reinterpret_cast<fftwf_complex*>(out));
    // scaling
    for (int i = 0; i < n0_ * n1_; ++i) {
        out[i] *= 1.0 / (n0_ * n1_);
    }
}

FFTMany::FFTMany(int n0, int howmany) : n0_(n0), n1_(1), howmany_(howmany)
{
    int rank = 1;
    int n[] = {n0};
    int istride = 1;
    int idist = n0;
    int ostride = 1;
    int odist = n0;
    plan_forward_ = fftwf_plan_many_dft(rank, n, howmany,
                                        0, NULL, istride, idist,
                                        0, NULL, ostride, odist,
                                        FFTW_FORWARD, FFTW_ESTIMATE);
    plan_inverse_ = fftwf_plan_many_dft(rank, n, howmany,
                                        0, NULL, istride, idist,
                                        0, NULL, ostride, odist,
                                        FFTW_BACKWARD, FFTW_ESTIMATE);
}

FFTMany::FFTMany(int n0, int n1, int howmany) : n0_(n0), n1_(n1), howmany_(howmany)
{
    int rank = 2;
    int n[] = {n0, n1};
    int istride = 1;
    int idist = n0 * n1;
    int ostride = 1;
    int odist = n0 * n1;
    plan_forward_ = fftwf_plan_many_dft(rank, n, howmany,
                                        0, NULL, istride, idist,
                                        0, NULL, ostride, odist,
                                        FFTW_FORWARD, FFTW_ESTIMATE);
    plan_inverse_ = fftwf_plan_many_dft(rank, n, howmany,
                                        0, NULL, istride, idist,
                                        0, NULL, ostride, odist,
                                        FFTW_BACKWARD, FFTW_ESTIMATE);
}

FFTMany::~FFTMany()
{
    fftwf_destroy_plan(plan_forward_);
    fftwf_destroy_plan(plan_inverse_);
}

void FFTMany::forward(const std::complex<float>* in, std::complex<float>* out) const
{
    fftwf_execute_dft(plan_forward_,
                      reinterpret_cast<fftwf_complex*>(const_cast<std::complex<float>*>(in)),
                      reinterpret_cast<fftwf_complex*>(out));
}

void FFTMany::inverse(const std::complex<float>* in, std::complex<float>* out) const
{
    fftwf_execute_dft(plan_inverse_,
                      reinterpret_cast<fftwf_complex*>(const_cast<std::complex<float>*>(in)),
                      reinterpret_cast<fftwf_complex*>(out));
    // scaling
    for (int i = 0; i < n0_ * n1_ * howmany_; ++i) {
        out[i] *= 1.0 / (n0_ * n1_);
    }
}
