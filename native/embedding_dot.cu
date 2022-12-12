#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "cuda_fp16.h"
#include "embedding_dot.hh"

inline void throw_if_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

template <typename scalar_t, typename accum_t>
__global__ void embedding_dot_forward(const scalar_t* __restrict__ embedding1,
                                      const scalar_t* __restrict__ embedding2,
                                      const uint32_t* __restrict__ indices,
                                      scalar_t* __restrict__ output,
                                      uint32_t num_features,
                                      uint32_t num_indices, uint32_t a_size,
                                      uint32_t b_size) {
    accum_t accum = 0;

    uint32_t index = threadIdx.y + blockIdx.x * blockDim.y;

    if (index < num_indices) {
        uint32_t embedding1_index = indices[2 * index + 0];
        uint32_t embedding2_index = indices[2 * index + 1];

        if (embedding1_index >= a_size || embedding2_index >= b_size) {
            output[index] = 0;
            return;
        }

        for (uint32_t featureDim = threadIdx.x; featureDim < num_features;
             featureDim += blockDim.x) {
            accum +=
                static_cast<accum_t>(
                    embedding1[embedding1_index * num_features + featureDim]) *
                static_cast<accum_t>(
                    embedding2[embedding2_index * num_features + featureDim]);
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            accum += __shfl_down_sync(0xffffffff, accum, offset);
        }

        if (threadIdx.x == 0) {
            // This is the warp leader
            output[index] = accum;
        }
    }
}

template <typename scalar_t, typename accum_t>
void apply_embedding_dot_forward(cudaStream_t stream, void** buffers,
                                 const char* opaque, std::size_t opaque_len) {
    const uint32_t* sizes = reinterpret_cast<const uint32_t*>(opaque);
    uint32_t num_features = sizes[0];
    uint32_t num_indices = sizes[1];
    uint32_t a_size = sizes[2];
    uint32_t b_size = sizes[3];

    const scalar_t* embedding1 = reinterpret_cast<const scalar_t*>(buffers[0]);
    const scalar_t* embedding2 = reinterpret_cast<const scalar_t*>(buffers[1]);
    const uint32_t* indices = reinterpret_cast<const uint32_t*>(buffers[2]);
    scalar_t* output = reinterpret_cast<scalar_t*>(buffers[3]);

    dim3 block = dim3(32, 8);
    int grid = (num_indices + 7) / 8;

    embedding_dot_forward<scalar_t, accum_t>
        <<<grid, block, 0, stream>>>(embedding1, embedding2, indices, output,
                                     num_features, num_indices, a_size, b_size);

    throw_if_cuda_error(cudaGetLastError());
}

void half_embedding_dot_forward(cudaStream_t stream, void** buffers,
                                const char* opaque, std::size_t opaque_len) {
    apply_embedding_dot_forward<__half, float>(stream, buffers, opaque,
                                               opaque_len);
}

void float_embedding_dot_forward(cudaStream_t stream, void** buffers,
                                 const char* opaque, std::size_t opaque_len) {
    apply_embedding_dot_forward<float, float>(stream, buffers, opaque,
                                              opaque_len);
}

template <typename scalar_t, typename accum_t>
__global__ void embedding_dot_backward(const scalar_t* __restrict__ embedding1,
                                       const scalar_t* __restrict__ embedding2,
                                       const uint32_t* __restrict__ indices,
                                       const scalar_t* __restrict__ output_grad,
                                       accum_t* __restrict__ embedding1_grad,
                                       accum_t* __restrict__ embedding2_grad,
                                       uint32_t num_features,
                                       uint32_t num_indices, uint32_t a_size,
                                       uint32_t b_size) {
    uint32_t index = threadIdx.y + blockIdx.x * blockDim.y;
    uint32_t side = blockIdx.y;
    uint32_t source_size = a_size;
    uint32_t destination_size = b_size;

    if (side == 1) {
        source_size = b_size;
        destination_size = a_size;
    }

    if (index < num_indices) {
        accum_t index_grad = output_grad[index];

        uint32_t source_index = indices[2 * index + side];
        uint32_t destination_index = indices[2 * index + 1 - side];

        if (source_index >= source_size ||
            destination_index >= destination_size) {
            return;
        }

        const scalar_t* source_location =
            (side ? embedding2 : embedding1) + source_index * num_features;
        accum_t* target_location = (side ? embedding1_grad : embedding2_grad) +
                                   destination_index * num_features;

        for (int32_t featureDim = threadIdx.x; featureDim < num_features;
             featureDim += blockDim.x) {
            accum_t source_value = source_location[featureDim];
            accum_t value = source_value * index_grad;
            atomicAdd(target_location + featureDim, value);
        }
    }
}

template <typename scalar_t, typename accum_t>
void apply_embedding_dot_backward(cudaStream_t stream, void** buffers,
                                  const char* opaque, std::size_t opaque_len) {
    const uint32_t* sizes = reinterpret_cast<const uint32_t*>(opaque);
    uint32_t num_features = sizes[0];
    uint32_t num_indices = sizes[1];
    uint32_t num_a = sizes[2];
    uint32_t num_b = sizes[3];

    const scalar_t* embedding1 = reinterpret_cast<const scalar_t*>(buffers[0]);
    const scalar_t* embedding2 = reinterpret_cast<const scalar_t*>(buffers[1]);
    const uint32_t* indices = reinterpret_cast<const uint32_t*>(buffers[2]);
    const scalar_t* output_grad = reinterpret_cast<const scalar_t*>(buffers[3]);
    accum_t* embedding1_grad = reinterpret_cast<accum_t*>(buffers[4]);
    accum_t* embedding2_grad = reinterpret_cast<accum_t*>(buffers[5]);

    dim3 block = dim3(32, 8);
    dim3 grid = dim3((num_indices + 7) / 8, 2);

    throw_if_cuda_error(cudaMemsetAsync(
        embedding1_grad, 0, sizeof(accum_t) * num_a * num_features, stream));
    throw_if_cuda_error(cudaMemsetAsync(
        embedding2_grad, 0, sizeof(accum_t) * num_b * num_features, stream));

    embedding_dot_backward<scalar_t, accum_t><<<grid, block, 0, stream>>>(
        embedding1, embedding2, indices, output_grad, embedding1_grad,
        embedding2_grad, num_features, num_indices, num_a, num_b);

    throw_if_cuda_error(cudaGetLastError());
}

void half_embedding_dot_backward(cudaStream_t stream, void** buffers,
                                 const char* opaque, std::size_t opaque_len) {
    apply_embedding_dot_backward<__half, float>(stream, buffers, opaque,
                                                opaque_len);
}

void float_embedding_dot_backward(cudaStream_t stream, void** buffers,
                                  const char* opaque, std::size_t opaque_len) {
    apply_embedding_dot_backward<float, float>(stream, buffers, opaque,
                                               opaque_len);
}
