#pragma once

#include <utility>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

constexpr dim3 BlockSize1 = {256, 1, 1};
constexpr dim3 BlockSize2 = {16, 16, 1};

inline dim3 block_size1(dim3 size)
{
    return {min(size.x, BlockSize1.x),
            min(size.y, BlockSize1.y),
            min(size.z, BlockSize1.z)};
}

inline dim3 block_size2(dim3 size)
{
    return {min(size.x, BlockSize2.x),
            min(size.y, BlockSize2.y),
            min(size.z, BlockSize2.z)};
}

inline dim3 grid_size(dim3 size, dim3 block)
{
    return dim3 {(size.x + block.x - 1) / block.x,
                 (size.y + block.y - 1) / block.y,
                 (size.z + block.z - 1) / block.z};
}

inline std::pair<dim3, dim3> grid_block_size1(dim3 size)
{
    dim3 block = block_size1(size);
    dim3 grid = grid_size(size, block);
    return {grid, block};
}

inline std::pair<dim3, dim3> grid_block_size2(dim3 size)
{
    dim3 block = block_size2(size);
    dim3 grid = grid_size(size, block);
    return {grid, block};
}

namespace gpu {

using thrust::device_vector;
using complex64  = thrust::complex<float>;

template <class T>
inline const T* to_ptr(const thrust::device_vector<T>& dv) {
    return thrust::raw_pointer_cast(dv.data());
}

template <class T>
inline T* to_ptr(thrust::device_vector<T>& dv) {
    return thrust::raw_pointer_cast(dv.data());
}

} // namespace gpu
