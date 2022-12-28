# GPU-accelerated compressive holography

The source code for compressive holography using FISTA on CUDA-enabled GPUs.

## Dependencies

- [CUDA](https://developer.nvidia.com/cuda-toolkit)
- [FFTW](http://www.fftw.org/)
- [CMake](https://cmake.org/)

## Build

```sh
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Usage

Sample hologram data are in the `data` directory.
`hologram_1234.png` is used for TV regularization, and `hologram_points.png` is used for L1 regularization.

Sample hologram reconstruction using L1 or TV regularization on GPU:
```sh
./test_cu l1 [number of iterations] [regularizer weight]
./test_cu tv [number of iterations] [regularizer weight]
```

Sample hologram reconstruction using L1 or TV regularization on CPU:
```sh
./test l1 [number of iterations] [regularizer weight]
./test tv [number of iterations] [regularizer weight]
```

## Author

- Yutaka Endo, Kanazawa University, endo@se.kanazawa-u.ac.jp

## Reference

- Yutaka Endo, Tomoyoshi Shimobaba, Takashi Kakue, and Tomoyoshi Ito, "GPU-accelerated compressive holography," Opt. Express 24, 8437-8445 (2016) [link](https://doi.org/10.1364/OE.24.008437)
