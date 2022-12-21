#pragma once

#include <vector>
#include <algorithm>
#include <numeric>

template <class T>
struct ForwardModel {
    virtual ~ForwardModel(){};
    virtual void forward(const std::vector<T>& in, std::vector<T>& out) = 0;
    virtual void adjoint(const std::vector<T>& in, std::vector<T>& out) = 0;
};

template <class T>
struct ProxLoss {
    virtual ~ProxLoss(){};
    virtual T operator()(const std::vector<T>& x) = 0;
    virtual void prox(const std::vector<T>& in, std::vector<T>& out, T t) = 0;
};

template <class T>
struct ProxLoss <std::complex<T>> {
    virtual ~ProxLoss(){};
    virtual T operator()(const std::vector<std::complex<T>>& x) = 0;
    virtual void prox(const std::vector<std::complex<T>>& in, std::vector<std::complex<T>>& out, T t) = 0;
};

template <class T>
T square_error(const std::vector<T>& x, const std::vector<T>& y)
{
    auto sqe = [](T x, T y) {
        auto d = x - y;
        return d * d;
    };
    return std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0, std::plus<T>(), sqe);
}

template <class T>
T square_error(const std::vector<std::complex<T>>& x, const std::vector<std::complex<T>>& y)
{
    auto sqe = [](std::complex<T> x, std::complex<T> y) {
        auto d = x - y;
        return d.real() * d.real() + d.imag() * d.imag();
    };
    return std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0, std::plus<T>(), sqe);
}

template <class T>
void fista(const std::vector<T>& y, std::vector<T>& x, ForwardModel<T>& model, ProxLoss<T>& g, int nit, double stepsize, double weight, bool verbose = true)
{
    std::vector<T> x_prev = x;
    std::vector<T> z = x;
    std::vector<T> grad(x.size());

    auto t = 1.0;
    auto t_prev = 1.0;

    for (int k = 0; k < nit; ++k) {
        std::copy(x.begin(), x.end(), x_prev.begin());

        // gradient descent
        model.forward(z, grad);
        for (int i = 0; i < y.size(); ++i) {
            grad[i] -= y[i];
        }
        model.adjoint(grad, grad);
        for (int i = 0; i < x.size(); ++i) {
            x[i] = z[i] - static_cast<T>(stepsize) * grad[i];
        }

        // proximal operator
        g.prox(x, x, weight * stepsize);

        // Nesterov's acceleration
        t_prev = t;
        t = 0.5 * (1 + sqrt(1 + 4 * t_prev * t_prev));
        T c = (t_prev - 1) / t;
        for (int i = 0; i < z.size(); ++i) {
            z[i] = x[i] + c * (x[i] - x_prev[i]);
        }

        // cost function
        if (verbose) {
            model.forward(x, grad);
            auto loss1 = square_error(y, grad) * 0.5;
            auto loss2 = g(x) * weight;
            printf("%d\t%.3e\t%.3e\t%.3e\n", k, loss1 + loss2, loss1, loss2);
        }
    }
}