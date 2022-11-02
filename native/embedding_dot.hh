#pragma once

#include <cuda_runtime.h>

#include <cstdint>

void half_embedding_dot_forward(cudaStream_t stream, void** buffers,
                                const char* opaque, std::size_t opaque_len);

void float_embedding_dot_forward(cudaStream_t stream, void** buffers,
                                 const char* opaque, std::size_t opaque_len);

void half_embedding_dot_backward(cudaStream_t stream, void** buffers,
                                 const char* opaque, std::size_t opaque_len);

void float_embedding_dot_backward(cudaStream_t stream, void** buffers,
                                  const char* opaque, std::size_t opaque_len);
