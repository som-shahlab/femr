#pragma once

#include <unsupported/Eigen/CXX11/Tensor>

#include "absl/types/span.h"

double compute_c_statistic(const Eigen::Tensor<double, 1>& times,
                           const Eigen::Tensor<bool, 1>& is_censor,
                           const Eigen::Tensor<double, 1>& time_bins,
                           const Eigen::Tensor<double, 2>& hazards);

std::vector<double> compute_calibration(const std::vector<double>& probs,
                                        const std::vector<bool>& is_censor,
                                        size_t num_intervals);