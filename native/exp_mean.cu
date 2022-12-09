#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "exp_mean.hh"

using namespace nvcuda;

constexpr int WIDTH = 32;
constexpr int HEIGHT = 8;
constexpr int DIM_SIZE = 16;
constexpr int WARP_SIZE = 32;

static_assert(WARP_SIZE == WIDTH, "Must be the same per logic of code");

constexpr int WARPS_PER_BLOCK = 16;

constexpr int MULT_SIZE = 8;

constexpr float log_2_factor =
    0.693147180559945309417232121458176568075500134360255254;

inline void throw_if_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

__device__ inline uint32_t divide(uint32_t x, uint32_t val, uint32_t shift,
                                  uint32_t mult) {
    if (val == 1) {
        return x;
    }
    uint32_t q = __umulhi(x, mult);
    uint32_t t = ((x - q) >> 1) + q;
    return (t >> (shift - 1));
}

__device__ inline uint32_t modulus(uint32_t x, uint32_t val, uint32_t shift,
                                   uint32_t mult) {
    uint32_t divisor = divide(x, val, shift, mult);
    return x - divisor * val;
}

__global__ void exp_mean_with_grad(
    const __half *__restrict__ a, const __half *__restrict__ b,
    const uint32_t *__restrict__ offsets, const float *__restrict__ defaults,
    const uint32_t *__restrict__ indices, const float *__restrict__ values,
    float *__restrict__ out, float *__restrict__ a_grad,
    float *__restrict__ b_grad, uint32_t n, uint32_t m, uint32_t k,
    uint32_t m_shift, uint32_t m_mult) {
    wmma::fragment<wmma::matrix_a, HEIGHT, WIDTH, DIM_SIZE, __half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_a, HEIGHT, WIDTH, DIM_SIZE, __half,
                   wmma::col_major>
        a_frag_col;
    wmma::fragment<wmma::matrix_b, HEIGHT, WIDTH, DIM_SIZE, __half,
                   wmma::col_major>
        b_frag;
    wmma::fragment<wmma::matrix_b, HEIGHT, WIDTH, DIM_SIZE, __half,
                   wmma::row_major>
        b_frag_row;
    wmma::fragment<wmma::accumulator, HEIGHT, WIDTH, DIM_SIZE, float> c_frag;

    __shared__ __align__(32)
        uint32_t shared_last_index[WARPS_PER_BLOCK * HEIGHT];

    extern __shared__ __align__(32) __half shared_temp[];

    __shared__ __align__(
        32) float shared_constant[WARPS_PER_BLOCK * HEIGHT * WIDTH];

    uint32_t block_x =
        divide(blockIdx.x,
               (m + HEIGHT * WARPS_PER_BLOCK - 1) / (HEIGHT * WARPS_PER_BLOCK),
               m_shift, m_mult);
    uint32_t block_y =
        modulus(block_x + blockIdx.x,
                (m + HEIGHT * WARPS_PER_BLOCK - 1) / (HEIGHT * WARPS_PER_BLOCK),
                m_shift, m_mult);

    uint32_t i_base = block_x * HEIGHT * WARPS_PER_BLOCK;
    uint32_t j_base = block_y * HEIGHT * WARPS_PER_BLOCK;

    uint32_t i = i_base + threadIdx.y * HEIGHT;

    uint32_t *last_index = shared_last_index + threadIdx.y * HEIGHT;
    __half *temp = shared_temp + threadIdx.y * HEIGHT * DIM_SIZE * MULT_SIZE;
    float *constant = shared_constant + threadIdx.y * HEIGHT * WIDTH;

    uint32_t load_row = threadIdx.x / (WARP_SIZE / HEIGHT);
    uint32_t load_row_index = (threadIdx.x % (WARP_SIZE / HEIGHT));

    bool loader = (load_row_index == 0) && (i < n);
    if (loader) {
        uint32_t start = offsets[i + load_row];
        uint32_t end = offsets[i + load_row + 1];

        while (start != end) {
            uint32_t probe = (start + end) / 2;
            uint32_t value = indices[probe];

            if (value > j_base) {
                end = probe;
            } else {
                start = probe + 1;
            }
        }

        if (start > 0 && (indices[start - 1] == j_base)) {
            start--;
        }

        last_index[load_row] = start;
    }

    __syncwarp();

    const __half *a_offset = a + i * k;

    float local_total = 0;

    for (uint32_t mult = 0; mult < MULT_SIZE; mult += 2) {
        uint32_t j = j_base + mult * DIM_SIZE;
        if (i < n && j < m) {
            const __half *b_offset = b + j * k;

            for (uint32_t i_in = 0; i_in < HEIGHT; i_in++) {
                constant[i_in * WIDTH + threadIdx.x] = defaults[i + i_in];

                bool valid = false;
                uint32_t current_index = last_index[i_in] + threadIdx.x;

                if (current_index < offsets[i + i_in + 1]) {
                    uint32_t target_j = indices[current_index] - j;

                    if (target_j < WIDTH) {
                        valid = true;
                        constant[i_in * WIDTH + target_j] =
                            values[current_index];
                    }
                }

                uint32_t valid_count = __ballot_sync(0xffffffff, valid);

                if (threadIdx.x == 0) {
                    uint32_t count = __popc(valid_count);
                    last_index[i_in] += count;
                }
            }

            wmma::load_matrix_sync(c_frag, constant, WIDTH,
                                   wmma::mem_row_major);

            for (uint32_t d = 0; d < k; d += DIM_SIZE) {
                wmma::load_matrix_sync(a_frag, a_offset + d, k);
                wmma::load_matrix_sync(b_frag, b_offset + d, k);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            for (uint32_t t = 0; t < c_frag.num_elements; t++) {
                float value = exp2f(c_frag.x[t]);
                c_frag.x[t] = value;
                local_total += value / (8 * 16 * 8);
            }

            wmma::store_matrix_sync(constant, c_frag, WIDTH,
                                    wmma::mem_row_major);

            for (uint32_t i_in = 0; i_in < HEIGHT; i_in++) {
                __half val = __float2half(constant[i_in * WIDTH + threadIdx.x] *
                                          log_2_factor);
                temp[i_in * DIM_SIZE * MULT_SIZE + mult * DIM_SIZE +
                     threadIdx.x] = val;
            }
        }
    }

    if (i < n) {
        float *a_grad_offset = a_grad + i * k;
        for (uint32_t d = 0; d < k; d += WIDTH) {
            wmma::fill_fragment(c_frag, 0);

            for (uint32_t mult = 0; mult < MULT_SIZE; mult++) {
                uint32_t j = j_base + mult * DIM_SIZE;

                if (j >= m) {
                    continue;
                }

                const __half *b_offset = b + j * k;

                wmma::load_matrix_sync(a_frag, temp + mult * DIM_SIZE,
                                       DIM_SIZE * MULT_SIZE);

                wmma::load_matrix_sync(b_frag_row, b_offset + d, k);

                wmma::mma_sync(c_frag, a_frag, b_frag_row, c_frag);
            }

            for (uint32_t t = 0; t < c_frag.num_elements; t++) {
                c_frag.x[t] /= (n * m);
            }

            wmma::store_matrix_sync(constant, c_frag, WIDTH,
                                    wmma::mem_row_major);

            for (uint32_t i_in = 0; i_in < HEIGHT; i_in++) {
                atomicAdd(a_grad_offset + i_in * k + d + threadIdx.x,
                          constant[i_in * WIDTH + threadIdx.x]);
            }
        }
    }

    __syncthreads();

    uint32_t j = j_base + threadIdx.y * HEIGHT;

    if (j < m) {
        float *b_grad_offset = b_grad + j * k;
        for (uint32_t d = 0; d < k; d += WIDTH) {
            wmma::fill_fragment(c_frag, 0);

            for (uint32_t mult = 0; mult < MULT_SIZE; mult++) {
                uint32_t i = i_base + mult * DIM_SIZE;

                if (i >= n) {
                    continue;
                }

                __half *temp_ptr =
                    shared_temp + mult * WARPS_PER_BLOCK * DIM_SIZE * MULT_SIZE;

                const __half *a_offset = a + i * k;

                wmma::load_matrix_sync(a_frag_col,
                                       temp_ptr + threadIdx.y * HEIGHT,
                                       DIM_SIZE * MULT_SIZE);

                wmma::load_matrix_sync(b_frag_row, a_offset + d, k);

                wmma::mma_sync(c_frag, a_frag_col, b_frag_row, c_frag);
            }

            for (uint32_t t = 0; t < c_frag.num_elements; t++) {
                c_frag.x[t] /= (n * m);
            }

            wmma::store_matrix_sync(constant, c_frag, WIDTH,
                                    wmma::mem_row_major);

            for (uint32_t i_in = 0; i_in < HEIGHT; i_in++) {
                atomicAdd(b_grad_offset + i_in * k + d + threadIdx.x,
                          constant[i_in * WIDTH + threadIdx.x]);
            }
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_total += __shfl_down_sync(0xffffffff, local_total, offset);
    }

    if (threadIdx.x == 0) {
        float result =
            atomicAdd(out, local_total * ((float)(8 * 16 * 8) / (n * m)));
    }
}

void half_exp_mean_with_grad(cudaStream_t stream, void **buffers,
                             const char *opaque, std::size_t opaque_len) {
    const uint32_t *sizes = reinterpret_cast<const uint32_t *>(opaque);
    uint32_t n = sizes[0];
    uint32_t m = sizes[1];
    uint32_t k = sizes[2];
    uint32_t m_shift = sizes[3];
    uint32_t m_mult = sizes[4];

    assert(k % WIDTH == 0);
    assert(n % HEIGHT == 0);
    assert(m % WIDTH == 0);

    const __half *a = reinterpret_cast<const __half *>(buffers[0]);
    const __half *b = reinterpret_cast<const __half *>(buffers[1]);
    const uint32_t *offsets = reinterpret_cast<const uint32_t *>(buffers[2]);
    const float *defaults = reinterpret_cast<const float *>(buffers[3]);
    const uint32_t *indices = reinterpret_cast<const uint32_t *>(buffers[4]);
    const float *values = reinterpret_cast<const float *>(buffers[5]);

    float *out = reinterpret_cast<float *>(buffers[6]);
    float *a_grad = reinterpret_cast<float *>(buffers[7]);
    float *b_grad = reinterpret_cast<float *>(buffers[8]);

    const int splits = HEIGHT * WARPS_PER_BLOCK;
    const int num_n_blocks = ((n + splits - 1) / splits);
    const int num_m_blocks = ((m + splits - 1) / splits);
    int numBlocks = num_n_blocks * num_m_blocks;
    dim3 threadsPerBlock(WARP_SIZE, WARPS_PER_BLOCK);

    throw_if_cuda_error(cudaMemsetAsync(out, 0, sizeof(*out) * 1, stream));
    throw_if_cuda_error(
        cudaMemsetAsync(a_grad, 0, sizeof(*a_grad) * n * k, stream));
    throw_if_cuda_error(
        cudaMemsetAsync(b_grad, 0, sizeof(*b_grad) * m * k, stream));

    int num_bytes =
        sizeof(__half) * WARPS_PER_BLOCK * HEIGHT * DIM_SIZE * MULT_SIZE;

    throw_if_cuda_error(cudaFuncSetAttribute(
        exp_mean_with_grad, cudaFuncAttributeMaxDynamicSharedMemorySize,
        num_bytes));

    exp_mean_with_grad<<<numBlocks, threadsPerBlock, num_bytes, stream>>>(
        a, b, offsets, defaults, indices, values, out, a_grad, b_grad, n, m, k,
        m_shift, m_mult);
}
