#pragma once

#include <cuda_runtime.h>

#include <cstdint>

void half_exp_mean_with_grad(cudaStream_t stream, void** buffers,
                             const char* opaque, std::size_t opaque_len);
