#pragma once

#include <cuda_runtime.h>

#include <cstdint>

void float_gather_scatter(cudaStream_t stream, void** buffers,
                          const char* opaque, std::size_t opaque_len);