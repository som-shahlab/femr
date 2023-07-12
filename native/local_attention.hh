#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

struct local_attention_info;

void half_local_attention_forward(cudaStream_t stream, void** buffers,
                                  const char* opaque, std::size_t opaque_len);

void half_local_attention_backward(cudaStream_t stream, void** buffers,
                                   const char* opaque, std::size_t opaque_len);

std::vector<uint32_t> get_attention_shape(uint32_t b, uint32_t n, uint32_t k,
                                          uint32_t w, bool causal);

const local_attention_info* create_attention_info(uint32_t b, uint32_t n,
                                                  uint32_t k, uint32_t w,
                                                  bool causal);

void free_attention_info(const local_attention_info* info);
