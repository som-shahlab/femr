#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

using namespace nvcuda;

constexpr int DIM_SIZE = 16;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 16;

static_assert(WARPS_PER_BLOCK == DIM_SIZE);  // Current algorithm assumes this

constexpr int LAUNCH_SIZE = 64;

static_assert(LAUNCH_SIZE % 4 == 0);  // Current algorithm assumes this

constexpr int NUM_C = 4;

// TODO: Allow more values of K
constexpr int K = 64;
constexpr float K_FACTOR = 1.0 / 8;  // 1 / sqrt(K)

static_assert(NUM_C * DIM_SIZE == K);

inline void throw_if_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

// For hiding bogus warnings
template <typename T>
void dummy_use(T a) {
    (void)a;
}

__device__ half2 vector_half_max(half2 a, half2 b) {
    float2 fa, fb, fr;
    fa = __half22float2(a);
    fb = __half22float2(b);

    fr.x = fmaxf(fa.x, fb.x);
    fr.y = fmaxf(fa.y, fb.y);

    return __float22half2_rn(fr);
}

__device__ half half_max(half a, half b) {
    float fa, fb, fr;
    fa = __half2float(a);
    fb = __half2float(b);

    fr = fmaxf(fa, fb);
    return __float2half_rn(fr);
}

struct alignas(8) i_info {
    uint32_t i;
    uint32_t start_index;
    uint32_t end_index;
};

struct alignas(8) j_info {
    const uint32_t *indexes;
    uint32_t j;
    uint32_t num_indexes;
    bool requires_atomic;
};

struct alignas(8) row_info {
    const uint2 *infos;
    const i_info *i_data;
    const j_info *j_data;

    uint32_t num_i;
    uint32_t num_j;
    uint32_t num_infos;
};

struct alignas(8) local_attention_info {
    const row_info *row_data;
    uint32_t num_bytes;

    uint32_t num_rows;

    uint32_t b;
    uint32_t n;
    uint32_t k;
    uint32_t w;
    bool causal;

    bool requires_any_atomic;
};

template <typename T>
T *alloc_and_copy(T *&dest, const T *to_copy, uint32_t size) {
    T *result = dest;
    dest += size;

    memcpy(result, to_copy, sizeof(T) * size);

    return result;
}

unsigned char *alloc_shared(uint32_t size) {
    unsigned char *result;
    throw_if_cuda_error(cudaMallocManaged(&result, size));
    return result;
}

template <typename T>
void convert_to_pointer(const std::vector<T> &data, const T *&ptr,
                        uint32_t &size) {
    T *result = new T[data.size()];
    memcpy(result, data.data(), sizeof(T) * data.size());
    ptr = result;
    size = data.size();
}

void free_attention_info(const local_attention_info *info) {
    throw_if_cuda_error(cudaFree((void *)info));
}

const local_attention_info *convert_to_cuda(const local_attention_info *info) {
    uint32_t num_rows = info->num_rows;
    uint32_t num_infos = 0;
    uint32_t num_i = 0;
    uint32_t num_j = 0;
    uint32_t num_indexes = 0;

    for (uint32_t i = 0; i < info->num_rows; i++) {
        const row_info &row = info->row_data[i];
        num_infos += row.num_infos;
        num_i += row.num_i;
        num_j += row.num_j;

        for (uint32_t j = 0; j < row.num_j; j++) {
            const j_info &j_i = row.j_data[j];
            num_indexes += j_i.num_indexes;
        }
    }

    std::map<uint32_t, std::set<uint32_t>> rows_for_j;
    for (uint32_t i = 0; i < info->num_rows; i++) {
        const row_info &r = info->row_data[i];
        for (uint32_t j = 0; j < r.num_j; j++) {
            const j_info &j_i = r.j_data[j];
            rows_for_j[j_i.j].insert(i);
        }
    }

    bool requires_any_atomic = false;

    for (const auto &entry : rows_for_j) {
        if (entry.second.size() > 1) {
            requires_any_atomic = true;
        }
    }

    size_t max_size = 1024 * 1024 * 1024;
    size_t total_size = max_size;

    // CUDA allocates on 256 byte boundaries
    char *start = (char *)(nullptr) + 256;
    void *dummy = start;

    auto helper = [&](size_t size, size_t number) {
        void *offset = std::align(256, size * number, dummy, total_size);
        total_size -= size * number;
        dummy = (void *)((char *)dummy + size * number);

        assert(offset != nullptr);

        return (size_t)((char *)offset - start);
    };

    size_t result_offset = helper(sizeof(*info), 1);
    size_t row_offset = helper(sizeof(*info->row_data), num_rows);
    size_t info_offset = helper(sizeof(*info->row_data->infos), num_infos);
    size_t i_offset = helper(sizeof(*info->row_data->i_data), num_i);
    size_t j_offset = helper(sizeof(*info->row_data->j_data), num_j);
    size_t indexes_offset =
        helper(sizeof(*info->row_data->j_data->indexes), num_indexes);

    uint32_t num_bytes = max_size - total_size;

    unsigned char *data = alloc_shared(num_bytes);

    local_attention_info *result_ptr =
        (local_attention_info *)(data + result_offset);
    row_info *result_rows = (row_info *)(data + row_offset);
    uint2 *infos = (uint2 *)(data + info_offset);
    i_info *i_s = (i_info *)(data + i_offset);
    j_info *j_s = (j_info *)(data + j_offset);
    uint32_t *indexes = (uint32_t *)(data + indexes_offset);

    local_attention_info *result = alloc_and_copy(result_ptr, info, 1);
    result->num_bytes = num_bytes;

    row_info *rows =
        alloc_and_copy(result_rows, info->row_data, info->num_rows);
    result->row_data = rows;

    for (uint32_t i = 0; i < info->num_rows; i++) {
        const row_info &row = info->row_data[i];
        row_info &result_row = rows[i];

        result_row.infos = alloc_and_copy(infos, row.infos, row.num_infos);

        delete row.infos;

        result_row.i_data = alloc_and_copy(i_s, row.i_data, row.num_i);
        delete row.i_data;

        j_info *res_js = alloc_and_copy(j_s, row.j_data, row.num_j);
        result_row.j_data = res_js;
        for (uint32_t j = 0; j < row.num_j; j++) {
            const j_info &j_i = row.j_data[j];
            j_info &res_j = res_js[j];

            res_j.indexes =
                alloc_and_copy(indexes, j_i.indexes, j_i.num_indexes);
            if (rows_for_j[j_i.j].size() == 1) {
                res_j.requires_atomic = false;
            }
            delete j_i.indexes;
        }
        delete row.j_data;
    }
    delete info->row_data;
    delete info;

    result->requires_any_atomic = requires_any_atomic;

    cudaMemAdvise(result, result->num_bytes, cudaMemAdviseSetReadMostly, 0);

    return result;
}

template <typename T, typename F>
std::vector<typename std::result_of<F(uint32_t, T *)>::type> join_on_first(
    std::vector<T> &input, F func) {
    std::vector<typename std::result_of<F(uint32_t, T *)>::type> result;

    std::sort(std::begin(input), std::end(input));

    size_t start_index = 0;

    for (size_t i = 1; i < input.size(); i++) {
        if (input[i].first != input[start_index].first) {
            result.emplace_back(
                func(i - start_index, input.data() + start_index));
            start_index = i;
        }
    }

    if (input.size() != 0) {
        result.emplace_back(
            func(input.size() - start_index, input.data() + start_index));
    }

    return result;
}

const local_attention_info *create_attention_info(uint32_t b, uint32_t n,
                                                  uint32_t k, uint32_t w,
                                                  bool causal) {
    local_attention_info *result = new local_attention_info;
    result->b = b;
    result->n = n;
    result->k = k;
    result->w = w;
    result->causal = causal;
    std::vector<row_info> rows;
    std::vector<std::pair<uint32_t, uint32_t>> current_launch;

    auto flush = [&]() {
        assert(current_launch.size() <= LAUNCH_SIZE);

        if (current_launch.size() > 0) {
            std::vector<std::pair<uint32_t, uint32_t>> current_i_locations;
            std::vector<std::pair<uint32_t, uint32_t>> current_j_locations;
            std::vector<uint2> infos;

            for (size_t i = 0; i < current_launch.size(); i++) {
                const auto &item = current_launch[i];
                current_i_locations.emplace_back(item.first, i);
                current_j_locations.emplace_back(item.second, i);

                uint2 next;
                next.x = item.first;
                next.y = item.second;
                infos.push_back(next);
            }

            row_info row;
            assert(infos.size() <= LAUNCH_SIZE);

            convert_to_pointer(infos, row.infos, row.num_infos);

            convert_to_pointer(
                join_on_first(
                    current_i_locations,
                    [&](uint32_t n, std::pair<uint32_t, uint32_t> *entries) {
                        i_info next;
                        next.i = entries[0].first;
                        next.start_index = entries[0].second;
                        next.end_index = entries[n - 1].second + 1;
                        return next;
                    }),
                row.i_data, row.num_i);

            convert_to_pointer(
                join_on_first(
                    current_j_locations,
                    [&](uint32_t n, std::pair<uint32_t, uint32_t> *entries) {
                        j_info next;
                        next.j = entries[0].first;
                        std::vector<uint32_t> indices;
                        for (uint32_t i = 0; i < n; i++) {
                            indices.push_back(entries[i].second);
                        }
                        convert_to_pointer(indices, next.indexes,
                                           next.num_indexes);
                        next.requires_atomic = true;
                        return next;
                    }),
                row.j_data, row.num_j);

            rows.push_back(row);

            current_launch.clear();
        }
    };

    for (uint32_t i = 0; i < n; i += DIM_SIZE) {
        uint32_t start_j = max(0, (int)i - (int)w);
        uint32_t end_j;
        if (causal) {
            end_j = i + DIM_SIZE;
        } else {
            end_j = min(i + w, n) + DIM_SIZE;
        }
        uint32_t needed = end_j - start_j;

        assert(needed % DIM_SIZE == 0);

        uint32_t needed_blocks = needed / DIM_SIZE;

        assert(needed_blocks <= LAUNCH_SIZE);
        if (current_launch.size() + needed_blocks > LAUNCH_SIZE) {
            flush();
        }

        for (uint32_t j = start_j; j < end_j; j += DIM_SIZE) {
            current_launch.emplace_back(i, j);
        }
    }

    flush();

    convert_to_pointer(rows, result->row_data, result->num_rows);

    const local_attention_info *cuda_result = convert_to_cuda(result);

    return cuda_result;
}

std::vector<uint32_t> get_attention_shape(uint32_t b, uint32_t n, uint32_t k,
                                          uint32_t w, bool causal) {
    const local_attention_info *info =
        create_attention_info(b, n, k, w, causal);
    std::vector<uint32_t> result = {b, info->num_rows, LAUNCH_SIZE * DIM_SIZE,
                                    DIM_SIZE};
    free_attention_info(info);
    return result;
}

__global__ void __launch_bounds__(WARPS_PER_BLOCK *WARP_SIZE)
    local_attention_backward(const local_attention_info *__restrict__ info,
                             const __half *__restrict__ all_queries,
                             const __half *__restrict__ all_keys,
                             const __half *__restrict__ all_values,
                             const uint32_t *__restrict__ length_mask,
                             const __half *__restrict__ all_attention,
                             const __half *__restrict__ all_result_d,
                             __half *__restrict__ all_dq,
                             float *__restrict__ all_dk,
                             float *__restrict__ all_dv) {
    wmma::fragment<wmma::matrix_a, DIM_SIZE, DIM_SIZE, DIM_SIZE, __half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_a, DIM_SIZE, DIM_SIZE, DIM_SIZE, __half,
                   wmma::col_major>
        a_frag_col;
    wmma::fragment<wmma::matrix_b, DIM_SIZE, DIM_SIZE, DIM_SIZE, __half,
                   wmma::col_major>
        b_frag[NUM_C];
    wmma::fragment<wmma::matrix_b, DIM_SIZE, DIM_SIZE, DIM_SIZE, __half,
                   wmma::row_major>
        b_frag_row;
    wmma::fragment<wmma::accumulator, DIM_SIZE, DIM_SIZE, DIM_SIZE, float>
        c_frag[NUM_C];
    wmma::fragment<wmma::accumulator, DIM_SIZE, DIM_SIZE, DIM_SIZE, float>
        d_frag;

    __shared__ __align__(
        32) float shared_constant[WARPS_PER_BLOCK * DIM_SIZE * DIM_SIZE];

    extern __shared__ __align__(32) __half shared_temp[];

    uint32_t n = info->n;
    uint32_t k = info->k;

    const __half *queries = all_queries + blockIdx.x * n * k;
    const __half *keys = all_keys + blockIdx.x * n * k;
    const __half *values = all_values + blockIdx.x * n * k;
    const __half *attention =
        all_attention +
        blockIdx.x * DIM_SIZE * DIM_SIZE * LAUNCH_SIZE * info->num_rows +
        blockIdx.y * DIM_SIZE * DIM_SIZE * LAUNCH_SIZE;
    const __half *result_d = all_result_d + blockIdx.x * n * k;

    __half *dq = all_dq + blockIdx.x * n * k;
    float *dk = all_dk + blockIdx.x * n * k;
    float *dv = all_dv + blockIdx.x * n * k;

    float *constant = shared_constant + threadIdx.y * DIM_SIZE * DIM_SIZE;

    const row_info &row = info->row_data[blockIdx.y];

    // We start by computing the derivative with respect to the attention
    // (stored in shared_temp) and to the values (stored in dv)

    for (uint32_t j_index = threadIdx.y; j_index < row.num_j;
         j_index += WARPS_PER_BLOCK) {
        const j_info &j_i = row.j_data[j_index];
        uint32_t j = j_i.j;
        uint32_t j_doc = j & *length_mask;

        for (uint32_t c = 0; c < NUM_C; c++) {
            uint32_t d = c * DIM_SIZE;
            wmma::fill_fragment(c_frag[c], 0);
            wmma::load_matrix_sync(b_frag[c], values + j * k + d, k);
        }

        for (uint32_t i_subindex = 0; i_subindex < j_i.num_indexes;
             i_subindex++) {
            uint32_t index = j_i.indexes[i_subindex];
            uint32_t i = row.infos[index].x;
            uint32_t i_doc = i & *length_mask;

            if (i_doc != j_doc) {
                continue;
            }

            wmma::load_matrix_sync(a_frag_col, attention + index * DIM_SIZE,
                                   LAUNCH_SIZE * DIM_SIZE);

            wmma::fill_fragment(d_frag, 0);
            for (uint32_t c = 0; c < NUM_C; c++) {
                uint32_t d = c * DIM_SIZE;
                wmma::load_matrix_sync(b_frag_row, result_d + i * k + d, k);
                wmma::load_matrix_sync(a_frag, result_d + i * k + d, k);

                wmma::mma_sync(c_frag[c], a_frag_col, b_frag_row, c_frag[c]);

                wmma::mma_sync(d_frag, a_frag, b_frag[c], d_frag);
            }

            wmma::store_matrix_sync(constant, d_frag, DIM_SIZE,
                                    wmma::mem_row_major);

            for (uint32_t i_in = threadIdx.x / 16; i_in < DIM_SIZE; i_in += 2) {
                float current_val =
                    constant[i_in * DIM_SIZE + (threadIdx.x % 16)];

                shared_temp[i_in * LAUNCH_SIZE * DIM_SIZE + index * DIM_SIZE +
                            (threadIdx.x % 16)] = current_val * K_FACTOR;
            }
        }

        if (j_i.requires_atomic) {
            for (uint32_t c = 0; c < NUM_C; c++) {
                uint32_t d = c * DIM_SIZE;
                wmma::store_matrix_sync(constant, c_frag[c], DIM_SIZE,
                                        wmma::mem_row_major);

                for (uint32_t i_in = threadIdx.x / 16; i_in < DIM_SIZE;
                     i_in += 2) {
                    float current_val =
                        constant[i_in * DIM_SIZE + (threadIdx.x % 16)];

                    atomicAdd(dv + (j + i_in) * k + d + (threadIdx.x % 16),
                              current_val);
                }
            }
        } else {
            for (uint32_t c = 0; c < NUM_C; c++) {
                uint32_t d = c * DIM_SIZE;
                wmma::store_matrix_sync(dv + j * k + d, c_frag[c], k,
                                        wmma::mem_row_major);
            }
        }
    }

    __syncthreads();

    // Given the derivative with respect to the attention (shared_temp), we can
    // now compute the derivative with respect to the attention's input using
    // the standard softmax matrix derivative
    // https://github.com/google/jax/blob/0860c2476781a2a62d27d7be8a969f27391ab987/jax/_src/nn/functions.py#L372
    // is the exact math involved here

    uint32_t i_in = threadIdx.y;

    for (uint32_t i_index = 0; i_index < row.num_i; i_index++) {
        const i_info &i_inf = row.i_data[i_index];
        uint32_t i = i_inf.i;
        uint32_t i_doc = i & *length_mask;

        // We start by computing the (y * x_dot).sum() term in the above
        // equation Need to use a float here to perform sums in high precision
        float current_sum = 0;

        for (uint32_t index = i_inf.start_index + threadIdx.x / 8;
             index < i_inf.end_index; index += 4) {
            uint32_t j = row.infos[index].y;
            uint32_t j_doc = j & *length_mask;

            if (j_doc != i_doc) {
                continue;
            }

            const __half2 &a =
                *(const __half2 *)(attention + i_in * LAUNCH_SIZE * DIM_SIZE +
                                   index * DIM_SIZE + (threadIdx.x % 8) * 2);
            // Add to running totals
            __half2 &current_val =
                *(__half2 *)(shared_temp + i_in * LAUNCH_SIZE * DIM_SIZE +
                             index * DIM_SIZE + (threadIdx.x % 8) * 2);

            current_val = __hmul2(current_val, a);
            current_sum += __low2float(current_val);
            current_sum += __high2float(current_val);
        }

        // Butterfly reduction so that every thread in the warp has the correct
        // sum
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            current_sum += __shfl_xor_sync(0xffffffff, current_sum, offset);
        }

        __half2 half_sum = __float2half2_rn(-current_sum);

        // Now that we have that sum, we can compute y * (x_dot - (y *
        // x_dot).sum())

        for (uint32_t index = i_inf.start_index + threadIdx.x / 8;
             index < i_inf.end_index; index += 4) {
            uint32_t j = row.infos[index].y;
            uint32_t j_doc = j & *length_mask;

            if (j_doc != i_doc) {
                continue;
            }

            const __half2 &a =
                *(const __half2 *)(attention + i_in * LAUNCH_SIZE * DIM_SIZE +
                                   index * DIM_SIZE + (threadIdx.x % 8) * 2);

            __half2 &val =
                *(__half2 *)(shared_temp + i_in * LAUNCH_SIZE * DIM_SIZE +
                             index * DIM_SIZE + (threadIdx.x % 8) * 2);

            val = __hfma2(a, half_sum, val);
        }
    }

    __syncthreads();

    // We now have the softmax derivatives, so we can now compute the derivates
    // with respect to the query and result

    for (uint32_t i_index = threadIdx.y; i_index < row.num_i; i_index += 4) {
        const i_info &i_inf = row.i_data[i_index];
        uint32_t i_doc = i_inf.i & *length_mask;

        for (uint32_t c = 0; c < NUM_C; c++) {
            wmma::fill_fragment(c_frag[c], 0);
        }

        for (uint32_t index = i_inf.start_index; index < i_inf.end_index;
             index++) {
            uint32_t j = row.infos[index].y;
            uint32_t j_doc = j & *length_mask;
            if (j_doc != i_doc) {
                continue;
            }

            // Now we compute the result
            wmma::load_matrix_sync(a_frag, shared_temp + index * DIM_SIZE,
                                   LAUNCH_SIZE * DIM_SIZE);

            for (uint32_t c = 0; c < NUM_C; c++) {
                uint32_t d = c * DIM_SIZE;
                wmma::load_matrix_sync(b_frag_row, keys + j * k + d, k);
                wmma::mma_sync(c_frag[c], a_frag, b_frag_row, c_frag[c]);
            }
        }

        uint32_t i = i_inf.i;
        for (uint32_t c = 0; c < NUM_C; c++) {
            uint32_t d = c * DIM_SIZE;

            wmma::store_matrix_sync(constant, c_frag[c], DIM_SIZE,
                                    wmma::mem_row_major);

            for (uint32_t i_in = threadIdx.x / 16; i_in < DIM_SIZE; i_in += 2) {
                float current_val =
                    constant[i_in * DIM_SIZE + (threadIdx.x % 16)];

                dq[(i + i_in) * k + d + (threadIdx.x % 16)] =
                    (__half)(current_val);
            }
        }
    }

    for (uint32_t j_index = threadIdx.y; j_index < row.num_j;
         j_index += WARPS_PER_BLOCK) {
        const j_info &j_i = row.j_data[j_index];
        uint32_t j = j_i.j;
        uint32_t j_doc = j & *length_mask;

        for (uint32_t c = 0; c < NUM_C; c++) {
            wmma::fill_fragment(c_frag[c], 0);
        }

        for (uint32_t i_subindex = 0; i_subindex < j_i.num_indexes;
             i_subindex++) {
            uint32_t index = j_i.indexes[i_subindex];
            uint32_t i = row.infos[index].x;
            uint32_t i_doc = i & *length_mask;

            if (i_doc != j_doc) {
                continue;
            }

            wmma::load_matrix_sync(a_frag_col, shared_temp + index * DIM_SIZE,
                                   LAUNCH_SIZE * DIM_SIZE);

            for (uint32_t c = 0; c < NUM_C; c++) {
                uint32_t d = c * DIM_SIZE;

                wmma::load_matrix_sync(b_frag_row, queries + i * k + d, k);
                wmma::mma_sync(c_frag[c], a_frag_col, b_frag_row, c_frag[c]);
            }
        }

        if (j_i.requires_atomic) {
            for (uint32_t c = 0; c < NUM_C; c++) {
                uint32_t d = c * DIM_SIZE;
                wmma::store_matrix_sync(constant, c_frag[c], DIM_SIZE,
                                        wmma::mem_row_major);

                for (uint32_t i_in = threadIdx.x / 16; i_in < DIM_SIZE;
                     i_in += 2) {
                    float current_val =
                        constant[i_in * DIM_SIZE + (threadIdx.x % 16)];

                    atomicAdd(dk + (j + i_in) * k + d + (threadIdx.x % 16),
                              current_val);
                }
            }
        } else {
            for (uint32_t c = 0; c < NUM_C; c++) {
                uint32_t d = c * DIM_SIZE;
                wmma::store_matrix_sync(dk + j * k + d, c_frag[c], k,
                                        wmma::mem_row_major);
            }
        }
    }
}

__global__ void __launch_bounds__(WARPS_PER_BLOCK *WARP_SIZE)
    local_attention_forward(const local_attention_info *__restrict__ info,
                            const __half *__restrict__ all_queries,
                            const __half *__restrict__ all_keys,
                            const __half *__restrict__ all_values,
                            const uint32_t *__restrict__ length_mask,
                            __half *__restrict__ all_attention,
                            __half *__restrict__ all_result) {
    wmma::fragment<wmma::matrix_a, DIM_SIZE, DIM_SIZE, DIM_SIZE, __half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, DIM_SIZE, DIM_SIZE, DIM_SIZE, __half,
                   wmma::col_major>
        b_frag;
    wmma::fragment<wmma::matrix_b, DIM_SIZE, DIM_SIZE, DIM_SIZE, __half,
                   wmma::row_major>
        b_frag_row;
    wmma::fragment<wmma::accumulator, DIM_SIZE, DIM_SIZE, DIM_SIZE, float>
        c_frag[NUM_C];

    __shared__ __align__(
        32) float shared_constant[WARPS_PER_BLOCK * DIM_SIZE * DIM_SIZE];

    extern __shared__ __align__(32) __half shared_temp[];

    uint32_t n = info->n;
    uint32_t k = info->k;
    uint32_t w = info->w;

    const __half *queries = all_queries + blockIdx.x * n * k;
    const __half *keys = all_keys + blockIdx.x * n * k;
    const __half *values = all_values + blockIdx.x * n * k;
    __half *attention =
        all_attention +
        blockIdx.x * DIM_SIZE * DIM_SIZE * LAUNCH_SIZE * info->num_rows +
        blockIdx.y * DIM_SIZE * DIM_SIZE * LAUNCH_SIZE;

    __half *result = all_result + blockIdx.x * n * k;

    float *constant = shared_constant + threadIdx.y * DIM_SIZE * DIM_SIZE;

    const row_info &row = info->row_data[blockIdx.y];

    for (uint32_t index = threadIdx.y; index < row.num_infos;
         index += WARPS_PER_BLOCK) {
        uint2 pos = row.infos[index];
        uint32_t i = pos.x;
        uint32_t j = pos.y;

        uint32_t i_doc = i & *length_mask;
        uint32_t j_doc = j & *length_mask;

        if (i_doc != j_doc) {
            continue;
        }

        wmma::fill_fragment(c_frag[0], 0);

        for (uint32_t d = 0; d < k; d += DIM_SIZE) {
            wmma::load_matrix_sync(a_frag, queries + i * k + d, k);
            wmma::load_matrix_sync(b_frag, keys + j * k + d, k);
            wmma::mma_sync(c_frag[0], a_frag, b_frag, c_frag[0]);
        }

        wmma::store_matrix_sync(constant, c_frag[0], DIM_SIZE,
                                wmma::mem_row_major);

        for (uint32_t i_in = threadIdx.x / 16; i_in < DIM_SIZE; i_in += 2) {
            float current_val = constant[i_in * DIM_SIZE + (threadIdx.x % 16)];

            shared_temp[i_in * LAUNCH_SIZE * DIM_SIZE + index * DIM_SIZE +
                        (threadIdx.x % 16)] = current_val / sqrt((float)k);
        }
    }

    __syncthreads();

    uint32_t i_in = threadIdx.y;

    for (uint32_t i_index = 0; i_index < row.num_i; i_index++) {
        const i_info &i_inf = row.i_data[i_index];
        uint32_t i = i_inf.i;
        uint32_t i_doc = i & *length_mask;

        __half2 current_maximum = __half2half2(-INFINITY);

        for (uint32_t index = i_inf.start_index + threadIdx.x / 8;
             index < i_inf.end_index; index += 4) {
            uint2 pos = row.infos[index];
            uint32_t j = pos.y;
            uint32_t j_doc = j & *length_mask;

            if (i_doc != j_doc) {
                continue;
            }

            // Add to running totals
            __half2 &current_val =
                *(__half2 *)(shared_temp + i_in * LAUNCH_SIZE * DIM_SIZE +
                             index * DIM_SIZE + (threadIdx.x % 8) * 2);
            __half2 copy = current_val;
            uint32_t precise_i = i + i_in;
            uint32_t precise_j = j + (threadIdx.x % 8) * 2;

            uint32_t precise_start = max(0, (int)precise_i - (int)w);
            uint32_t precise_end;
            if (info->causal) {
                precise_end = precise_i;
            } else {
                precise_end = precise_i + w;
            }

            if (precise_j > precise_end || precise_j < precise_start) {
                current_val.x = -INFINITY;
            }

            if ((precise_j + 1) > precise_end ||
                (precise_j + 1) < precise_start) {
                current_val.y = -INFINITY;
            }

            current_maximum = vector_half_max(current_maximum, current_val);
        }

        __half actual_max =
            half_max(__high2half(current_maximum), __low2half(current_maximum));

#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            actual_max = half_max(
                actual_max, __shfl_xor_sync(0xffffffff, actual_max, offset));
        }

        __half2 final_max = __half2half2(actual_max);

        float current_sum = 0;
        for (uint32_t index = i_inf.start_index + threadIdx.x / 8;
             index < i_inf.end_index; index += 4) {
            uint2 pos = row.infos[index];
            uint32_t j = pos.y;
            uint32_t j_doc = j & *length_mask;

            if (i_doc != j_doc) {
                continue;
            }

            __half2 &val =
                *(__half2 *)(shared_temp + i_in * LAUNCH_SIZE * DIM_SIZE +
                             index * DIM_SIZE + (threadIdx.x % 8) * 2);
            val = h2exp(__hsub2(val, final_max));
            current_sum += __low2float(val);
            current_sum += __high2float(val);
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            current_sum += __shfl_xor_sync(0xffffffff, current_sum, offset);
        }

        __half2 half_sum = __float2half2_rn(1 / current_sum);

        for (uint32_t index = i_inf.start_index + threadIdx.x / 8;
             index < i_inf.end_index; index += 4) {
            uint2 pos = row.infos[index];
            uint32_t j = pos.y;
            uint32_t j_doc = j & *length_mask;

            if (i_doc != j_doc) {
                continue;
            }

            __half2 &val =
                *(__half2 *)(shared_temp + i_in * LAUNCH_SIZE * DIM_SIZE +
                             index * DIM_SIZE + (threadIdx.x % 8) * 2);

            val = __hmul2(val, half_sum);

            __half2 *attention_ptr =
                (__half2 *)(attention + i_in * LAUNCH_SIZE * DIM_SIZE +
                            index * DIM_SIZE + (threadIdx.x % 8) * 2);

            *attention_ptr = val;
        }
    }

    __syncthreads();

    for (uint32_t i_index = threadIdx.y; i_index < row.num_i;
         i_index += WARPS_PER_BLOCK) {
        const i_info &i_inf = row.i_data[i_index];
        uint32_t i = i_inf.i;
        uint32_t i_doc = i & *length_mask;

        for (uint32_t c = 0; c < NUM_C; c++) {
            wmma::fill_fragment(c_frag[c], 0);
        }

        for (uint32_t index = i_inf.start_index; index < i_inf.end_index;
             index++) {
            uint32_t j = row.infos[index].y;
            uint32_t j_doc = j & *length_mask;

            if (i_doc != j_doc) {
                continue;
            }

            // Now we compute the result
            wmma::load_matrix_sync(a_frag, shared_temp + index * DIM_SIZE,
                                   LAUNCH_SIZE * DIM_SIZE);

            for (uint32_t c = 0; c < NUM_C; c++) {
                uint32_t d = c * DIM_SIZE;
                wmma::load_matrix_sync(b_frag_row, values + j * k + d, k);
                wmma::mma_sync(c_frag[c], a_frag, b_frag_row, c_frag[c]);
            }
        }

        for (uint32_t c = 0; c < NUM_C; c++) {
            uint32_t d = c * DIM_SIZE;

            wmma::store_matrix_sync(constant, c_frag[c], DIM_SIZE,
                                    wmma::mem_row_major);

            for (uint32_t i_in = threadIdx.x / 16; i_in < DIM_SIZE; i_in += 2) {
                float current_val =
                    constant[i_in * DIM_SIZE + (threadIdx.x % 16)];

                result[(i + i_in) * k + d + (threadIdx.x % 16)] =
                    (__half)(current_val);
            }
        }
    }
}

void half_local_attention_forward(cudaStream_t stream, void **buffers,
                                  const char *opaque, std::size_t opaque_len) {
    const local_attention_info *info =
        *reinterpret_cast<const local_attention_info *const *>(opaque);

    uint32_t b = info->b;
    uint32_t n = info->n;
    uint32_t k = info->k;
    uint32_t w = info->w;

    // TODO: Support more values of k
    assert(k == K);

    assert(n % DIM_SIZE == 0);
    assert(k % DIM_SIZE == 0);
    assert(w % DIM_SIZE == 0);

    // Hide bogus warnings
    dummy_use(n);
    dummy_use(k);
    dummy_use(w);

    const __half *queries = reinterpret_cast<const __half *>(buffers[0]);
    const __half *keys = reinterpret_cast<const __half *>(buffers[1]);
    const __half *values = reinterpret_cast<const __half *>(buffers[2]);
    const uint32_t *length_mask =
        reinterpret_cast<const uint32_t *>(buffers[3]);

    __half *attention = reinterpret_cast<__half *>(buffers[4]);
    __half *result = reinterpret_cast<__half *>(buffers[5]);

    dim3 numBlocks(b, info->num_rows);
    dim3 threadsPerBlock(WARP_SIZE, WARPS_PER_BLOCK);

    int needed_bytes = LAUNCH_SIZE * DIM_SIZE * DIM_SIZE * 2;

    throw_if_cuda_error(cudaFuncSetAttribute(
        local_attention_forward, cudaFuncAttributeMaxDynamicSharedMemorySize,
        needed_bytes));

    int device = -1;
    throw_if_cuda_error(cudaGetDevice(&device));

    throw_if_cuda_error(cudaMemPrefetchAsync(info, info->num_bytes, device));

    local_attention_forward<<<numBlocks, threadsPerBlock, needed_bytes,
                              stream>>>(info, queries, keys, values,
                                        length_mask, attention, result);
}

void half_local_attention_backward(cudaStream_t stream, void **buffers,
                                   const char *opaque, std::size_t opaque_len) {
    const local_attention_info *info =
        *reinterpret_cast<const local_attention_info *const *>(opaque);

    uint32_t b = info->b;
    uint32_t n = info->n;
    uint32_t k = info->k;
    uint32_t w = info->w;

    // TODO: Support more values of k
    assert(k == K);

    assert(n % DIM_SIZE == 0);
    assert(k % DIM_SIZE == 0);
    assert(w % DIM_SIZE == 0);

    // Hide bogus warnings
    dummy_use(n);
    dummy_use(k);
    dummy_use(w);

    const __half *queries = reinterpret_cast<const __half *>(buffers[0]);
    const __half *keys = reinterpret_cast<const __half *>(buffers[1]);
    const __half *values = reinterpret_cast<const __half *>(buffers[2]);
    const uint32_t *length = reinterpret_cast<const uint32_t *>(buffers[3]);

    const __half *attention = reinterpret_cast<const __half *>(buffers[4]);
    const __half *result_d = reinterpret_cast<const __half *>(buffers[5]);

    __half *dq = reinterpret_cast<__half *>(buffers[6]);
    float *dk = reinterpret_cast<float *>(buffers[7]);
    float *dv = reinterpret_cast<float *>(buffers[8]);

    dim3 numBlocks(b, info->num_rows);
    dim3 threadsPerBlock(WARP_SIZE, WARPS_PER_BLOCK);

    int needed_bytes = LAUNCH_SIZE * DIM_SIZE * DIM_SIZE * 2;

    throw_if_cuda_error(cudaFuncSetAttribute(
        local_attention_backward, cudaFuncAttributeMaxDynamicSharedMemorySize,
        needed_bytes));

    int device = -1;
    throw_if_cuda_error(cudaGetDevice(&device));
    throw_if_cuda_error(cudaMemPrefetchAsync(info, info->num_bytes, device));

    if (info->requires_any_atomic) {
        throw_if_cuda_error(
            cudaMemsetAsync(dk, 0, sizeof(*dk) * n * k * b, stream));
        throw_if_cuda_error(
            cudaMemsetAsync(dv, 0, sizeof(*dv) * n * k * b, stream));
    }

    local_attention_backward<<<numBlocks, threadsPerBlock, needed_bytes,
                               stream>>>(info, queries, keys, values, length,
                                         attention, result_d, dq, dk, dv);
}
