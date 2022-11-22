#include "survival_metrics.hh"

#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

size_t get_size(size_t needed) {
    size_t candidate = 1;

    while (true) {
        candidate <<= 1;
        if (candidate + 1 >= needed) {
            return candidate + 1;
        }
    }
}

// Least Significant Bit of i having a value of 1
// #define LSB(i) ((i) & -(i))

size_t LSB(size_t i) { return ((i) & -(i)); }

// Returns the sum of the first i elements (indices 0 to i)
// Equivalent to range_sum(0, i)
int compute_prefix_sum(const std::vector<int>& A, size_t i) {
    int sum = A[0];
    for (; i != 0; i -= LSB(i)) sum += A[i];
    return sum;
}

// Add delta to element with index i (zero-based)
void add(std::vector<int>& A, size_t i, int delta) {
    if (i == 0) {
        A[0] += delta;
        return;
    }
    for (; i < A.size(); i += LSB(i)) A[i] += delta;
}

void init(std::vector<int>& A) {
    size_t needed = get_size(A.size());
    if (A.size() != needed) {
        A.resize(needed);
    }
    for (size_t i = 1; i < A.size(); ++i) {
        size_t j = i + LSB(i);
        if (j < A.size()) A[j] += A[i];
    }
}

double compute_c_statistic(const Eigen::Tensor<double, 1>& times,
                           const Eigen::Tensor<bool, 1>& is_censor,
                           const Eigen::Tensor<double, 1>& time_bins,
                           const Eigen::Tensor<double, 2>& hazards) {
    ssize_t num_elem = hazards.dimension(0);
    ssize_t num_times = hazards.dimension(1);

    if (num_elem != times.dimension(0) || num_elem != is_censor.dimension(0)) {
        throw std::runtime_error("Times and censor spans don't match");
    }

    if (num_times != time_bins.dimension(0)) {
        throw std::runtime_error(
            "Number of time bins doesn't match the cdf shape");
    }

    std::vector<size_t> time_indices(num_elem);
    std::iota(std::begin(time_indices), std::end(time_indices), 0);

    uint32_t seed = 97;

    auto rng = std::default_random_engine(seed);

    std::shuffle(std::begin(time_indices), std::end(time_indices), rng);

    std::stable_sort(std::begin(time_indices), std::end(time_indices),
                     [&](size_t a, size_t b) { return times(a) < times(b); });

    double last_time = -1;

    std::vector<size_t> current_dead;

    std::vector<size_t> location_map(num_elem);
    std::vector<std::pair<double, size_t>> current_sorted_indices;
    std::vector<int> prefix_sum(num_elem, 0);
    std::vector<bool> is_still_alive(num_elem, true);

    double total_weight = 0;
    double total_auroc = 0;

    auto order_remaining_by_hazard = [&](uint32_t current_bucket) {
        prefix_sum.clear();
        current_sorted_indices.clear();
        for (ssize_t i = 0; i < num_elem; i++) {
            double hazard = hazards(i, current_bucket);

            current_sorted_indices.push_back(std::make_pair(hazard, i));
        }

        std::shuffle(std::begin(current_sorted_indices),
                     std::end(current_sorted_indices), rng);

        std::stable_sort(std::begin(current_sorted_indices),
                         std::end(current_sorted_indices));

        location_map.clear();
        location_map.resize(current_sorted_indices.size());

        for (size_t i = 0; i < current_sorted_indices.size(); i++) {
            location_map[current_sorted_indices[i].second] = i;
            bool alive = is_still_alive[current_sorted_indices[i].second];
            prefix_sum.push_back(alive);
        }

        init(prefix_sum);
    };

    auto resolve = [&]() {
        if (current_dead.size() != 0) {
            size_t num_true = compute_prefix_sum(
                prefix_sum, current_sorted_indices.size() - 1);
            size_t num_false = current_dead.size();
            size_t num_correct = 0;
            for (size_t dead : current_dead) {
                num_correct +=
                    compute_prefix_sum(prefix_sum, location_map[dead]);
            }
            double auroc;
            if (num_true == 0) {
                auroc = 1;
            } else {
                auroc = (double)num_correct / num_true;
            }
            total_weight += num_false;
            total_auroc += auroc;
        }
    };

    ssize_t current_bucket_index = 0;
    order_remaining_by_hazard(0);

    for (const auto& index : time_indices) {
        double current_time = times[index];
        if (current_time != last_time) {
            resolve();
            while (current_bucket_index != (time_bins.size() - 1) &&
                   current_time >= time_bins(current_bucket_index + 1)) {
                current_bucket_index++;
                order_remaining_by_hazard(current_bucket_index);
            }

            last_time = current_time;
            current_dead.clear();
        }

        if (is_censor(index) == 0) {
            current_dead.push_back(index);
        }

        add(prefix_sum, location_map[index], -1);
        is_still_alive[index] = false;
    }

    resolve();

    return total_auroc / total_weight;
}

std::vector<double> compute_calibration(absl::Span<const double> probs,
                                        absl::Span<const bool> is_censor,
                                        size_t num_intervals) {
    std::vector<double> result(num_intervals);

    if (is_censor.size() != probs.size()) {
        throw std::runtime_error("Invalid sizes ...");
    }

    for (size_t i = 0; i < is_censor.size(); i++) {
        bool censor = is_censor[i];
        double prob = probs[i];

        if (censor) {
            double s = 1 - prob;
            double density = 1 / prob;
            for (size_t bin_index = 0; bin_index < num_intervals; bin_index++) {
                double start = (double)bin_index / num_intervals;
                double end = (double)(bin_index + 1) / num_intervals;

                if (s < start) {
                    result[bin_index] += (end - start) * density;
                } else if (s >= start && s < end) {
                    result[bin_index] += (end - s) * density;
                } else if (s >= end) {
                    continue;
                }
            }
        } else {
            for (size_t bin_index = 0; bin_index < num_intervals; bin_index++) {
                double start = (double)bin_index / num_intervals;
                double end = (double)(bin_index + 1) / num_intervals;

                if (prob < start) {
                    continue;
                } else if (prob >= start && prob < end) {
                    result[bin_index] += 1;
                } else if (prob >= end) {
                    continue;
                }
            }
        }
    }

    return result;
}
