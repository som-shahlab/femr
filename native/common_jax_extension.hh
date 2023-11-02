#pragma once
#include <cmath>
#include <vector>

constexpr int DIM_SIZE = 16;
constexpr int LAUNCH_SIZE = 64;

inline std::vector<uint32_t> get_attention_shape(uint32_t b, uint32_t n, uint32_t k,
                                          uint32_t w, bool causal) {
    std::vector<std::pair<uint32_t, uint32_t>> current_launch;

    uint32_t num_rows = 0;
    for (uint32_t i = 0; i < n; i += DIM_SIZE) {
        uint32_t start_j = std::max(0, (int)i - (int)w);
        uint32_t end_j;
        if (causal) {
            end_j = i + DIM_SIZE;
        } else {
            end_j = std::min(i + w, n) + DIM_SIZE;
        }
        uint32_t needed = end_j - start_j;

        assert(needed % DIM_SIZE == 0);

        uint32_t needed_blocks = needed / DIM_SIZE;

        assert(needed_blocks <= LAUNCH_SIZE);
        if (current_launch.size() + needed_blocks > LAUNCH_SIZE) {
            if (current_launch.size() > 0) {
                num_rows++;
                current_launch.clear();
            }
        }

        for (uint32_t j = start_j; j < end_j; j += DIM_SIZE) {
            current_launch.emplace_back(i, j);
        }
    }

    if (current_launch.size() > 0) {
        num_rows++;
        current_launch.clear();
    }

    std::vector<uint32_t> result = {b, num_rows, LAUNCH_SIZE * DIM_SIZE,
                                    DIM_SIZE};

    return result;
}

inline void convert_to_dense(void *out, const void **in) {
    float *actual_out = reinterpret_cast<float *>(out);

    int32_t n = *reinterpret_cast<const int64_t *>(in[0]);
    int32_t m = *reinterpret_cast<const int64_t *>(in[1]);

    const int32_t *offsets = reinterpret_cast<const int32_t *>(in[2]);
    const float *defaults = reinterpret_cast<const float *>(in[3]);
    const int32_t *indices = reinterpret_cast<const int32_t *>(in[4]);
    const float *values = reinterpret_cast<const float *>(in[5]);

    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < m; j++) {
            actual_out[i * m + j] = defaults[i];
        }
        for (int32_t offset = offsets[i]; offset < offsets[i + 1]; offset++) {
            actual_out[i * m + indices[offset]] = values[offset];
        }
    }
}
