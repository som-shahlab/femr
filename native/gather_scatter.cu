#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "cuda_fp16.h"
#include "gather_scatter.hh"

inline void throw_if_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

const int WARP_SIZE = 32;
const int NUM_INDICES_PER_WARP = 32;

template <typename scalar_t>
__global__ void gather_scatter_kernel(
    /*
        This is a relatively straightforward gather scatter kenel.
        The basic idea is that each warp is assigned NUM_INDICES_PER_WARP
       indices. Each warp iterates through those assigned indices 1 by 1,
       reading from the input and writing to the output.

        There are two complications here:
        - We reduce the number of writes by only writing when we encounter a new
       index
        - We have to perform atomic writes if there is overlap between warps.

        Indices that are out of range are dropped.
    */

    const scalar_t* __restrict__ input, const uint32_t* __restrict__ indices,
    scalar_t* __restrict__ output, uint32_t num_indices, uint32_t feature_size,
    uint32_t input_size, uint32_t output_size) {
    // The start of the NUM_INDICES_PER_WARP indices to process
    uint32_t start_index =
        (threadIdx.y + blockIdx.x * blockDim.y) * NUM_INDICES_PER_WARP;

    // The feature offsets to process. Note that each warp processes WARP_SIZE
    // features.
    uint32_t feature_index = blockIdx.y * WARP_SIZE;

    // A temporary that stores the current sum.
    scalar_t temp_scalar = 0;

    // The last index the prior warp will process.
    uint32_t last_warp_end;
    if (start_index == 0) {
        last_warp_end = output_size;
    } else {
        last_warp_end = indices[(start_index - 1) * 2 + 1];
    }

    // The first index the next warp will process.
    uint32_t next_warp_start;
    if (start_index + 32 >= num_indices) {
        next_warp_start = output_size;
    } else {
        next_warp_start = indices[(start_index + NUM_INDICES_PER_WARP) * 2 + 1];
    }

    // TODO: We probably want to unroll this loop a bit for extra speed.
    for (uint32_t index_offset = 0; index_offset < NUM_INDICES_PER_WARP;
         index_offset++) {
        uint32_t index = start_index + index_offset;
        if (index >= num_indices) {
            break;
        }

        uint32_t in_index = indices[index * 2 + 0];
        uint32_t out_index = indices[index * 2 + 1];

        if (in_index >= input_size || out_index >= output_size) {
            // Out of range so break.
            break;
        }

        uint32_t next_out_index;
        if (__builtin_expect(index == (num_indices - 1), 0)) {
            next_out_index = output_size;
        } else {
            next_out_index = indices[(index + 1) * 2 + 1];
        }

        // Read from the gather input
        temp_scalar +=
            input[in_index * feature_size + feature_index + threadIdx.x];

        bool is_last = (index_offset == NUM_INDICES_PER_WARP - 1) ||
                       (out_index != next_out_index);

        if (is_last) {
            // Last so we need to write

            // Note that this might need to be an atomic write
            bool requires_atomic =
                (last_warp_end == out_index) || (next_warp_start == out_index);
            uint32_t out_offset =
                out_index * feature_size + feature_index + threadIdx.x;

            if (requires_atomic) {
                atomicAdd(output + out_offset, temp_scalar);
            } else {
                output[out_offset] += temp_scalar;
            }
            temp_scalar = 0;
        }
    }
}

template <typename scalar_t>
void apply_gather_scatter(cudaStream_t stream, void** buffers,
                          const char* opaque, std::size_t opaque_len) {
    const uint32_t* sizes = reinterpret_cast<const uint32_t*>(opaque);
    uint32_t num_indices = sizes[0];
    uint32_t num_features = sizes[1];
    uint32_t input_size = sizes[2];
    uint32_t output_size = sizes[3];

    const scalar_t* input = reinterpret_cast<const scalar_t*>(buffers[0]);
    const uint32_t* indices = reinterpret_cast<const uint32_t*>(buffers[1]);
    scalar_t* output = reinterpret_cast<scalar_t*>(buffers[2]);

    const int NUM_WARPS = 8;
    dim3 block = dim3(WARP_SIZE, NUM_WARPS);
    uint32_t num_per_thread = NUM_WARPS * NUM_INDICES_PER_WARP;
    dim3 grid = dim3((num_indices + num_per_thread - 1) / num_per_thread,
                     num_features / WARP_SIZE);

    // Note that we need to initalize it to zero in case we need any atomic
    // writes. Without any atomic writes, we wouldn't need this init step.
    throw_if_cuda_error(cudaMemsetAsync(
        output, 0, sizeof(scalar_t) * output_size * num_features, stream));
    gather_scatter_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(input, indices, output, num_indices,
                                     num_features, input_size, output_size);

    throw_if_cuda_error(cudaGetLastError());
}

void float_gather_scatter(cudaStream_t stream, void** buffers,
                          const char* opaque, std::size_t opaque_len) {
    apply_gather_scatter<float>(stream, buffers, opaque, opaque_len);
}
