#pragma once

#include <random>
#include <vector>

template <typename T>
class ReservoirSampler {
   public:
    ReservoirSampler(size_t size) : total_size(size), int_dist(0, size - 1) {
        total_weight = 0;
    }

    void add(T sample, double weight, std::mt19937& rng) {
        total_weight += weight;
        if (samples.size() < total_size) {
            samples.push_back(sample);
            if (samples.size() == total_size) {
                j = dist(rng);
                p_none = 1;
            }
        } else {
            double prob = weight / total_weight;

            j -= prob * p_none;
            p_none = p_none * (1 - prob);

            if (j <= 0) {
                samples[int_dist(rng)] = sample;
                j = dist(rng);
                p_none = 1;
            }
        }
    }

    void combine(const ReservoirSampler& other, std::mt19937& rng,
                 double force_weight = 0) {
        double total_weight = force_weight;
        if (total_weight == 0) {
            total_weight = other.get_total_weight();
        }
        for (auto val : other.get_samples()) {
            add(val, total_weight / other.get_samples().size(), rng);
        }
    }

    const std::vector<T>& get_samples() const { return samples; }

    double get_total_weight() const { return total_weight; }

   private:
    size_t total_size;

    std::vector<T> samples;
    double total_weight;
    double j;
    double p_none;

    std::uniform_real_distribution<> dist;
    std::uniform_int_distribution<> int_dist;
};

class OnlineStatistics {
   public:
    void add_value(double weight, double value) {
        count += weight;
        double delta = value - mean;
        mean += delta * (weight / count);
        double delta2 = value - mean;

        M2 += weight * (delta * delta2);
    }

    void combine(const OnlineStatistics& other) {
        if (count == 0) {
            count = other.count;
            mean = other.mean;
            M2 = other.M2;
        } else if (other.count != 0) {
            double total = count + other.count;
            double delta = other.mean - mean;
            double new_mean = mean + delta * (other.count / total);
            double new_M2 =
                M2 + other.M2 + (delta * count) * (delta * other.count) / total;

            count = total;
            mean = new_mean;
            M2 = new_M2;
        }
    }

    double get_mean() const { return mean; }

    double get_stddev() const { return std::sqrt(M2 / count); }

   private:
    double count = 0;
    double mean = 0;
    double M2 = 0;
};
