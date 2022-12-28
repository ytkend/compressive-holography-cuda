#pragma once

#include <cuda/std/complex>
#include <thrust/device_vector.h>
#include "common.cuh"

namespace gpu {

template <class T>
struct ForwardModel {
    virtual ~ForwardModel(){};
    virtual void forward(const device_vector<T>& in, device_vector<T>& out) = 0;
    virtual void adjoint(const device_vector<T>& in, device_vector<T>& out) = 0;
};

template <class T>
struct ProxLoss {
    virtual ~ProxLoss(){};
    virtual T operator()(const device_vector<T>& x) = 0;
    virtual void prox(const device_vector<T>& in, device_vector<T>& out, T t) = 0;
};

template <class T>
struct ProxLoss <thrust::complex<T>> {
    virtual ~ProxLoss(){};
    virtual T operator()(const device_vector<thrust::complex<T>>& x) = 0;
    virtual void prox(const device_vector<thrust::complex<T>>& in, device_vector<thrust::complex<T>>& out, T t) = 0;
};

template <class T>
T square_error(const device_vector<T>& x, device_vector<T>& y)
{
    device_vector<T> d(x.size());
    thrust::transform(x.begin(), x.end(), y.begin(), d.begin(), thrust::minus<T>());
    return thrust::transform_reduce(d.begin(), d.end(),
                                    [] __device__ (T z) { return z * z; },
                                    0.0, thrust::plus<T>());
}

template <class T>
T square_error(const device_vector<thrust::complex<T>>& x, device_vector<thrust::complex<T>>& y)
{
    device_vector<thrust::complex<T>> d(x.size());
    thrust::transform(x.begin(), x.end(), y.begin(), d.begin(), thrust::minus<thrust::complex<T>>());
    return thrust::transform_reduce(d.begin(), d.end(),
                                    [] __device__ (thrust::complex<T> z) { return z.real() * z.real() + z.imag() * z.imag(); },
                                    0.0, thrust::plus<T>());
}

template <class T>
void fista(const device_vector<T>& y, device_vector<T>& x, ForwardModel<T>& model, ProxLoss<T>& g, int nit, double stepsize, double weight, bool verbose = true)
{
    device_vector<T> x_prev = x;
    device_vector<T> z = x;
    device_vector<T> grad(x.size());

    auto t = 1.0;
    auto t_prev = 1.0;

    for (int k = 0; k < nit; ++k) {
        thrust::copy(x.begin(), x.end(), x_prev.begin());

        // gradient descent
        model.forward(z, grad);
        thrust::transform(grad.begin(), grad.begin() + y.size(), y.begin(), grad.begin(), thrust::minus<T>());
        model.adjoint(grad, grad);
        thrust::transform(z.begin(), z.end(), grad.begin(), x.begin(),
                          [stepsize] __device__ (T z, T grad) { return z - stepsize * grad; });

        // proximal operator
        g.prox(x, x, weight * stepsize);

        // Nesterov's acceleration
        t_prev = t;
        t = 0.5 * (1 + sqrt(1 + 4 * t_prev * t_prev));
        T c = (t_prev - 1) / t;
        thrust::transform(x.begin(), x.end(), x_prev.begin(), z.begin(),
                          [c] __device__ (T x, T x_prev) { return x + c * (x - x_prev); });

        if (verbose) {
            model.forward(x, grad);
            auto loss1 = square_error(y, grad) * 0.5;
            auto loss2 = g(x) * weight;
            printf("%d\t%.3e\t%.3e\t%.3e\n", k, loss1 + loss2, loss1, loss2);
        }
    }
}

} // namespace gpu