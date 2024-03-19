#include "survival_metrics.hh"

#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "absl/strings/str_cat.h"

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
double compute_prefix_sum(const std::vector<double>& A, size_t i) {
    double sum = A[0];
    for (; i != 0; i -= LSB(i)) sum += A[i];
    return sum;
}

// Add delta to element with index i (zero-based)
void add(std::vector<double>& A, size_t i, double delta) {
    if (i == 0) {
        A[0] += delta;
        return;
    }
    for (; i < A.size(); i += LSB(i)) A[i] += delta;
}

void init(std::vector<double>& A) {
    size_t needed = get_size(A.size());
    if (A.size() != needed) {
        A.resize(needed);
    }
    for (size_t i = 1; i < A.size(); ++i) {
        size_t j = i + LSB(i);
        if (j < A.size()) A[j] += A[i];
    }
}

std::pair<double, Eigen::Tensor<double, 2>> compute_c_statistic(
    const Eigen::Tensor<double, 1>& times,
    const Eigen::Tensor<bool, 1>& is_censor,
    const Eigen::Tensor<double, 1>& time_bins,
    const Eigen::Tensor<double, 2>& hazards) {
	Eigen::Tensor<double, 1> dummy_weights(times.dimension(0));
	dummy_weights.setConstant(1);
	return compute_c_statistic_weighted(times, is_censor, time_bins, hazards, dummy_weights);
}

std::pair<double, Eigen::Tensor<double, 2>> compute_c_statistic_weighted(
    const Eigen::Tensor<double, 1>& times,
    const Eigen::Tensor<bool, 1>& is_censor,
    const Eigen::Tensor<double, 1>& time_bins,
    const Eigen::Tensor<double, 2>& hazards,
    const Eigen::Tensor<double, 1>& sample_weights
    ) {
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
    std::vector<double> prefix_sum(num_elem, 0);
    std::vector<bool> is_still_alive(num_elem, true);

    ssize_t num_events = 0;
    for (ssize_t i = 0; i < num_elem; i++) {
        if (!is_censor[i]) {
            num_events += 1;
        }
    }

    double total_auroc = 0;
    double total_weight = 0;

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
	    double weight = 0;
	    if (alive) {
		    weight = sample_weights[current_sorted_indices[i].second];
	    }

            prefix_sum.push_back(weight);
        }

        init(prefix_sum);
    };

    double surv = 1;

    std::vector<double> weights;

    std::vector<std::tuple<double, double, double>> survival_vals;

    auto resolve = [&]() {
        if (current_dead.size() != 0) {
            double num_true = compute_prefix_sum(
                prefix_sum, current_sorted_indices.size() - 1);
	    double num_false = 0;
            double num_correct = 0;
            for (size_t dead : current_dead) {
		num_false += sample_weights[dead];
                num_correct +=
                    sample_weights[dead] * compute_prefix_sum(prefix_sum, location_map[dead]);
            }

            double frac_died = num_false / (num_true + num_false);
            double next_surv = surv * (1 - frac_died);
            double death_rate = surv * frac_died;

            double weight = death_rate * next_surv;

            if (num_true == 0) {
                assert(weight == 0);
            } else {
                double auroc = (double)num_correct / (num_true * num_false);

                survival_vals.push_back(
                    std::make_tuple(auroc, last_time, weight));

                weights.push_back((weight * num_events) / num_false);

                total_weight += weight;
                total_auroc += weight * auroc;
            }

            surv = next_surv;
        }
    };

    ssize_t current_bucket_index = 0;
    order_remaining_by_hazard(0);

    for (size_t i = 0; i < time_indices.size(); i++) {
        size_t index = time_indices[i];
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

        add(prefix_sum, location_map[index], -sample_weights[index]);
        is_still_alive[index] = false;
    }

    resolve();

    Eigen::Tensor<double, 2> survival_plot_result(survival_vals.size(), 3);

    for (size_t i = 0; i < survival_vals.size(); i++) {
        auto entry = survival_vals[i];
        survival_plot_result(i, 0) = std::get<0>(entry);
        survival_plot_result(i, 1) = std::get<1>(entry);
        survival_plot_result(i, 2) = std::get<2>(entry);
    }

    double result = total_auroc / total_weight;

    return {result, survival_plot_result};
}

std::vector<double> apply_breslow(const Eigen::Tensor<double, 1>& times,
                                  const Eigen::Tensor<double, 1>& time_bins,
                                  const Eigen::Tensor<double, 2>& hazards,
                                  const Eigen::Tensor<double, 2>& breslow) {
    ssize_t num_elem = hazards.dimension(0);
    ssize_t num_times = hazards.dimension(1);

    ssize_t num_breslow = breslow.dimension(0);

    if (num_elem != times.dimension(0)) {
        throw std::runtime_error("Times and censor spans don't match");
    }

    if (num_times != time_bins.dimension(0)) {
        throw std::runtime_error(
            "Number of time bins doesn't match the cdf shape");
    }

    std::vector<double> breslow_times;
    breslow_times.reserve(num_breslow);
    for (ssize_t i = 0; i < num_breslow; i++) {
        breslow_times.push_back(breslow(i, 1));
    }

    std::vector<size_t> time_indices;

    for (ssize_t i = 0; i < num_times; i++) {
        auto iter =
            std::lower_bound(std::begin(breslow_times), std::end(breslow_times),
                             time_bins[i], std::less_equal<double>{});
        time_indices.push_back(iter - std::begin(breslow_times));
    }

    std::vector<double> result;
    result.reserve(num_elem);

    for (ssize_t i = 0; i < num_elem; i++) {
        double time = times[i];
        double total_hazard = 0;

        ssize_t current_time_bin = 0;

        while (current_time_bin < (num_times - 1) &&
               time > time_bins[current_time_bin + 1]) {
            size_t start = time_indices[current_time_bin];
            size_t end = time_indices[current_time_bin + 1];

            double total = breslow(end - 1, 0) - breslow(start - 1, 0);

            total_hazard += total * hazards(i, current_time_bin);
            current_time_bin++;
        }

        size_t start = time_indices[current_time_bin];
        auto iter =
            std::lower_bound(std::begin(breslow_times), std::end(breslow_times),
                             time, std::less_equal<double>{});
        size_t end = iter - std::begin(breslow_times);

        double total = breslow(end - 1, 0) - breslow(start - 1, 0);

        size_t offset = (iter == std::end(breslow_times)) ? 1 : 0;

        double bin_delta =
            breslow(end - offset, 0) - breslow(end - (1 + offset), 0);
        double bin_time =
            breslow_times[end - offset] - breslow_times[end - (1 + offset)];
        double rate = bin_delta / bin_time;

        total += rate * (time - breslow_times[end - 1]);

        total_hazard += total * hazards(i, current_time_bin);
        double prob = -std::expm1(-total_hazard);

        // Terrible hack to work around numeric problems ...
        if (prob >= 0.999) {
            prob = 0.999;
        }
        // if (prob == 1 && time != 0) {
        //    std::cout<<"What in the world .. " << total_hazard << " " << time
        //    << " " << hazards(i, current_time_bin) << " " << total <<
        //    std::endl;
        // }
        result.push_back(prob);
    }

    return result;
}

Eigen::Tensor<double, 2> estimate_breslow(
    const Eigen::Tensor<double, 1>& times,
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
    std::vector<bool> is_still_alive(num_elem, true);

    double current_sum = 0;
    std::vector<std::pair<double, double>> estimator;
    estimator.push_back(std::make_pair(0, 0));

    double current_denom;

    ssize_t current_bucket_index = 0;

    auto resolve = [&]() {
        if (current_dead.size() != 0) {
            double val = current_dead.size() * 1 / current_denom;

            for (size_t dead : current_dead) {
                current_denom -= hazards(dead, current_bucket_index);
            }

            current_sum += val;
            estimator.push_back(std::make_pair(current_sum, last_time));
        }
    };

    auto reset_denom = [&](uint32_t bucket_index) {
        current_denom = 0;

        for (ssize_t i = 0; i < num_elem; i++) {
            if (is_still_alive[i]) {
                current_denom += hazards(i, bucket_index);
            }
        }
    };

    reset_denom(0);

    for (size_t i = 0; i < time_indices.size(); i++) {
        size_t index = time_indices[i];
        double current_time = times[index];
        if (current_time != last_time) {
            resolve();
            while (current_bucket_index != (time_bins.size() - 1) &&
                   current_time >= time_bins(current_bucket_index + 1)) {
                current_bucket_index++;
                reset_denom(current_bucket_index);
            }

            last_time = current_time;
            current_dead.clear();
        }

        if (is_censor(index) == 0) {
            current_dead.push_back(index);
        } else {
            current_denom -= hazards(index, current_bucket_index);
        }
        is_still_alive[index] = false;
    }

    resolve();

    Eigen::Tensor<double, 2> result(estimator.size(), 2);

    for (size_t i = 0; i < estimator.size(); i++) {
        auto entry = estimator[i];
        result(i, 0) = entry.first;
        result(i, 1) = entry.second;
    }

    return result;
}

Eigen::Tensor<double, 2> estimate_optimal(
    const Eigen::Tensor<double, 1>& times,
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
    std::vector<bool> is_still_alive(num_elem, true);

    double current_sum = 0;
    std::vector<std::pair<double, double>> estimator;
    estimator.push_back(std::make_pair(0, 0));

    double current_denom;

    ssize_t current_bucket_index = 0;

    auto resolve = [&]() {
        if (current_dead.size() != 0) {
            double val = current_dead.size() * 1 / current_denom;

            for (size_t dead : current_dead) {
                current_denom -= hazards(dead, current_bucket_index);
            }

            current_sum += val;
            estimator.push_back(std::make_pair(current_sum, last_time));
        }
    };

    auto reset_denom = [&](uint32_t bucket_index) {
        current_denom = 0;

        for (ssize_t i = 0; i < num_elem; i++) {
            if (is_still_alive[i]) {
                current_denom += hazards(i, bucket_index);
            }
        }
    };

    reset_denom(0);

    for (size_t i = 0; i < time_indices.size(); i++) {
        size_t index = time_indices[i];
        double current_time = times[index];
        if (current_time != last_time) {
            resolve();
            while (current_bucket_index != (time_bins.size() - 1) &&
                   current_time >= time_bins(current_bucket_index + 1)) {
                current_bucket_index++;
                reset_denom(current_bucket_index);
            }

            last_time = current_time;
            current_dead.clear();
        }

        if (is_censor(index) == 0) {
            current_dead.push_back(index);
        } else {
            current_denom -= hazards(index, current_bucket_index);
        }
        is_still_alive[index] = false;
    }

    resolve();

    Eigen::Tensor<double, 2> result(estimator.size(), 2);

    for (size_t i = 0; i < estimator.size(); i++) {
        auto entry = estimator[i];
        result(i, 0) = entry.first;
        result(i, 1) = entry.second;
    }

    return result;
}

std::vector<double> compute_calibration(const std::vector<double>& probs,
                                        const std::vector<bool>& is_censor,
                                        size_t num_intervals) {
    std::vector<double> result(num_intervals);

    if (is_censor.size() != probs.size()) {
        throw std::runtime_error("Invalid sizes ...");
    }

    for (size_t i = 0; i < is_censor.size(); i++) {
        bool censor = is_censor[i];
        double prob = probs[i];

        if (censor) {
            double density = 1 / (1 - prob);
            double total = 0;

            for (size_t bin_index = 0; bin_index < num_intervals; bin_index++) {
                double start = (double)bin_index / num_intervals;
                double end = (double)(bin_index + 1) / num_intervals;

                double val = 0;

                if (prob < start) {
                    val = density * (end - start);
                } else if (prob >= start && prob < end) {
                    val = density * (end - prob);
                } else if (prob >= end) {
                    continue;
                }

                total += val;
                result[bin_index] += val;
            }

            if (abs(total - 1) > 0.01) {
                throw std::runtime_error(
                    absl::StrCat("This should never happen ", total, " ",
                                 density, " ", prob));
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

    for (double& element : result) {
        element /= probs.size();
    }

    return result;
}
