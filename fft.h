#pragma once

#include <fftw3.h>
#include <complex>
#include <vector>
#include <cassert>

class FFT
{
    int n0_, n1_;
    fftwf_plan plan_forward_;
    fftwf_plan plan_inverse_;

public:
    explicit FFT(int n0);
    FFT(int n0, int n1); // n1 is the fastest changing dimension of a transform
    ~FFT();

    void forward(const std::complex<float>* in, std::complex<float>* out) const;
    void inverse(const std::complex<float>* in, std::complex<float>* out) const;
};

class FFTMany
{
    int n0_, n1_, howmany_;
    fftwf_plan plan_forward_;
    fftwf_plan plan_inverse_;

public:
    FFTMany(int n0, int howmany);
    FFTMany(int n0, int n1, int howmany); // n1 is the fastest changing dimension of a transform
    ~FFTMany();

    void forward(const std::complex<float>* in, std::complex<float>* out) const;
    void inverse(const std::complex<float>* in, std::complex<float>* out) const;
};

inline int set_fft_threads(int n_threads)
{
    fftwf_init_threads();
    fftwf_plan_with_nthreads(n_threads);
    return fftwf_planner_nthreads();
}

inline void cleanup_fft_threads()
{
    fftwf_cleanup_threads();
}

template <typename T>
void swap_half(const T* in, T* out, int n, int first)
{
    int second = n - first;

    if (in == out) {
        std::vector<T> in_copy(in, in + n);
        for (int i = 0; i < n; ++i) {
            out[i] = (i < second) ? in_copy[i + first] : in_copy[i - second];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            out[i] = (i < second) ? in[i + first] : in[i - second];
        }
    }
}

template <typename T>
void swap_half(const T* in, T* out, int n0, int n1, int first0, int first1)
{
    int second0 = n0 - first0;
    int second1 = n1 - first1;

    if (in == out) {
        std::vector<T> in_copy(in, in + n0 * n1);
        for (int i0 = 0; i0 < n0; ++i0) {
            for (int i1 = 0; i1 < n1; ++i1) {
                int index0 = (i0 < second0) ? i0 + first0 : i0 - second0;
                int index1 = (i1 < second1) ? i1 + first1 : i1 - second1;
                out[i1 + i0 * n1] = in_copy[index1 + index0 * n1];
            }
        }
    } else {
        for (int i0 = 0; i0 < n0; ++i0) {
            for (int i1 = 0; i1 < n1; ++i1) {
                int index0 = (i0 < second0) ? i0 + first0 : i0 - second0;
                int index1 = (i1 < second1) ? i1 + first1 : i1 - second1;
                out[i1 + i0 * n1] = in[index1 + index0 * n1];
            }
        }
    }
}

// [0,1,2,3] -> [2,3,0,1]
// [0,1,2,3,4] -> [3,4,0,1,2]
template <typename T>
void fftshift(const T* in, T* out, int n)
{
    swap_half(in, out, n, (n + 1) / 2);
}

// [0,1,2,3] -> [2,3,0,1]
// [0,1,2,3,4] -> [2,3,4,0,1]
template <typename T>
void ifftshift(const T* in, T* out, int n)
{
    swap_half(in, out, n, n / 2);
}

template <typename T>
void fftshift(const T* in, T* out, int n0, int n1)
{
    swap_half(in, out, n0, n1, (n0 + 1) / 2, (n1 + 1) / 2);
}

template <typename T>
void ifftshift(const T* in, T* out, int n0, int n1)
{
    swap_half(in, out, n0, n1, n0 / 2, n1 / 2);
}

template <class T>
void pad(const T* in, T* out, int n, int before, int after, T val = 0)
{
    std::fill(out, out + before, val);
    std::copy(in, in + n, out + before);
    std::fill(out + before + n, out + before + n + after, val);
}

template <class T>
std::vector<T> pad(const T* in, int n, int before, int after, T val = 0)
{
    std::vector<T> out(n);
    pad(in, out.data(), n, before, after, val);
    return out;
}

template <class T>
void pad(const T* in, T* out, int n0, int n1, int before0, int before1, int after0, int after1, T val = 0)
{
    int istride = n1;
    int ostride = n1 + before1 + after1;

    std::fill(out, out + before0 * ostride, val);

    for (int i0 = before0; i0 < before0 + n0; ++i0) {
        std::fill(out + i0 * ostride,
                  out + i0 * ostride + before1, val);
        std::copy(in + (i0 - before0) * istride,
                  in + (i0 - before0) * istride + n1,
                  out + i0 * ostride + before1);
        std::fill(out + i0 * ostride + before1 + n1,
                  out + i0 * ostride + before1 + n1 + after1, val);
    }

    std::fill(out + (before0 + n0) * ostride,
              out + (before0 + n0 + after0) * ostride, val);
}

template <class T>
std::vector<T> pad(const T* in, int n0, int n1, int before0, int before1, int after0, int after1, T val = 0)
{
    std::vector<T> out((before0 + n0 + after0) * (before1 + n1 + after1));
    pad(in, out.data(), n0, n1, before0, before1, after0, after1, val);
    return out;
}

template <class T>
void crop(const T* in, T* out, int first, int last)
{
    std::copy(in + first, in + first + last, out);
}

template <class T>
std::vector<T> crop(const T* in, int first, int last)
{
    std::vector<T> out(last - first);
    crop(in, out.data(), first, last);
    return out;
}

template <class T>
void crop(const T* in, T* out, int n1, int first0, int first1, int last0, int last1)
{
    int istride = n1;
    int ostride = last1 - first1;

    for (int i0 = 0; i0 < last0 - first0; ++i0) {
        std::copy(in + (i0 + first0) * istride + first1,
                  in + (i0 + first0) * istride + first1 + ostride,
                  out + i0 * ostride);
    }
}

template <class T>
std::vector<T> crop(const std::vector<T>& in, int n1, int first0, int first1, int last0, int last1)
{
    std::vector<T> out((last0 - first0) * (last1 - first1));
    crop(in.data(), out.data(), n1, first0, first1, last0, last1);
    return out;
}

// template <class T>
// std::vector<T> pad(const std::vector<T>& in, int nx, int ny, int before_x, int before_y, int after_x, int after_y, T val)
// {
//     // check vector size
//     assert(in.size() == nx * ny);

//     int nx_o = nx + before_x + after_x;
//     int ny_o = ny + before_y + after_y;
//     std::vector<T> out(nx_o * ny_o);

//     for (int y = 0; y < ny_o; ++y) {
//         for (int x = 0; x < nx_o; ++x) {
//             if (y >= before_y && y < before_y + ny &&
//                 x >= before_x && x < before_x + nx) {
//                 out[x + y * nx_o] = in[(x - before_x) + (y - before_y) * nx];
//             } else {
//                 out[x + y * nx_o] = val;
//             }
//         }
//     }

//     return out;
// }

// template <class T>
// void crop(const T* in, T* out, int istride, int begin_x, int begin_y, int end_x, int end_y)
// {
//     int nx_o = end_x - begin_x;
//     int ny_o = end_y - begin_y;

//     for (int y = 0; y < ny_o; ++y) {
//         for (int x = 0; x < nx_o; ++x) {
//             out[x + y * nx_o] = in[(x + begin_x) + (y + begin_y) * istride];
//         }
//     }
// }
