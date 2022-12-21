# GPU-accelerated compressive holography

This repository containes the source code for compressive holography using FISTA on GPUs.

## Dependencies

- CUDA
- FFTW

## Build

```sh
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Usage


Reconstruction using L1 regularization on GPU
```sh
./test_cu l1 [number of iterations] [regularizer weight]
```

Reconstruction using TV regularization on GPU
```sh
./test_cu tv [number of iterations] [regularizer weight]
```

Reconstruction using L1 regularization on CPU
```sh
./test l1 [number of iterations] [regularizer weight]
```

Reconstruction using TV regularization on CPU
```sh
./test tv [number of iterations] [regularizer weight]
```

## Reference

- Yutaka Endo, Tomoyoshi Shimobaba, Takashi Kakue, and Tomoyoshi Ito, "GPU-accelerated compressive holography," Opt. Express 24, 8437-8445 (2016) [PDF](https://doi.org/10.1364/OE.24.008437)
