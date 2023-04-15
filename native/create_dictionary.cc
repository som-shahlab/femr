#include "create_dictionary.hh"

#include <cmath>
#include <nlohmann/json.hpp>
#include <queue>
#include <random>

#include "absl/container/flat_hash_map.h"
#include "clmbr_dictionary.hh"
#include "database.hh"
#include "flatmap.hh"
#include "stat_utils.hh"

using json = nlohmann::json;

// The number of samples to use for numeric values
constexpr int NUM_SAMPLES = 10000;

// The state for doing the dictionary calculation
struct DictionaryData {
    // The age mean and variance
    OnlineStatistics age_stats;

    // The count of times a code appears
    FlatMap<double> code_counts;

    // The count of times a code appears with ontology expansion
    FlatMap<double> hierarchical_code_counts;

    // The count of times each text value appears
    FlatMap<absl::flat_hash_map<uint32_t, double>> text_counts;

    // Random numeric samples for each lab value
    FlatMap<ReservoirSampler<float>> numeric_samples;

    // A random number generator
    std::mt19937 rng;
};

// Update the DictionaryData with another patient
void add_patient_to_dictionary(DictionaryData& data, const Patient& p,
                               Ontology& ontology, size_t num_patients,
                               const FlatMap<bool>& banned_codes) {
    // We want each patient to get roughly uniform weight, so scale by number of
    // patients and number of events.
    double weight = 1.0 / (num_patients * p.events.size());

    for (const auto& event : p.events) {
        // Remove banned codes
        if (banned_codes.find(event.code) != nullptr) {
            continue;
        }

        // Remove unique text
        if (event.value_type == ValueType::UNIQUE_TEXT) {
            continue;
        }

        // Add the start age to the age statistics
        data.age_stats.add_value(weight, event.start_age_in_minutes);

        switch (event.value_type) {
            case ValueType::NONE:
                // Update the counts
                for (uint32_t parent : ontology.get_all_parents(event.code)) {
                    *data.hierarchical_code_counts.find_or_insert(parent, 0) +=
                        weight;
                }
                *data.code_counts.find_or_insert(event.code, 0) += weight;
                break;

            case ValueType::NUMERIC: {
                // Add a numeric sample
                auto iter = data.numeric_samples.find(event.code);
                if (iter == nullptr) {
                    data.numeric_samples.insert(
                        event.code, ReservoirSampler<float>(NUM_SAMPLES));
                    iter = data.numeric_samples.find(event.code);
                }
                iter->add(event.numeric_value, weight, data.rng);
                break;
            }

            case ValueType::SHARED_TEXT:
                // Add a text sample
                (*data.text_counts.find_or_insert(
                    event.code, {}))[event.text_value] += weight;
                break;

            default:
                throw std::runtime_error("Invalid value type?");
        }
    }
}

// As part of the map-reduce, merge several dictionary datas
void merge_dictionary(DictionaryData& result, const DictionaryData& to_merge) {
    // Merge the age statistics
    result.age_stats.combine(to_merge.age_stats);

    // Accumulate the weights
    for (uint32_t code : to_merge.code_counts.keys()) {
        const double* weight = to_merge.code_counts.find(code);
        *result.code_counts.find_or_insert(code, 0) += *weight;
    }

    // Accumulate other weights
    for (uint32_t code : to_merge.hierarchical_code_counts.keys()) {
        const double* weight = to_merge.hierarchical_code_counts.find(code);
        *result.hierarchical_code_counts.find_or_insert(code, 0) += *weight;
    }

    // Accumulate the text counts
    for (uint32_t code : to_merge.text_counts.keys()) {
        const auto* text_entries = to_merge.text_counts.find(code);
        auto* target_text_entries = result.text_counts.find_or_insert(code, {});
        for (const auto& entry : *text_entries) {
            (*target_text_entries)[entry.first] += entry.second;
        }
    }

    // Accumulate the numeric samples
    for (uint32_t code : to_merge.numeric_samples.keys()) {
        const auto* samples = to_merge.numeric_samples.find(code);
        auto* target_samples = result.numeric_samples.find_or_insert(
            code, ReservoirSampler<float>(NUM_SAMPLES));
        target_samples->combine(*samples, result.rng);
    }
}

void create_dictionary(const std::string& input, const std::string& output) {
    boost::filesystem::path path(input);
    PatientDatabase database(path, true);

    // Prime the pump
    database.get_ontology().get_all_parents(0);

    // Remove all STANFORD_OBS codes for now ...
    FlatMap<bool> banned_codes;
    std::string_view banned_prefix = "STANFORD_OBS";

    uint32_t num_banned = 0;
    for (uint32_t code = 0; code < database.get_code_dictionary().size();
         code++) {
        std::string_view text_str = database.get_code_dictionary()[code];
        if (text_str.substr(0, banned_prefix.size()) == banned_prefix) {
            banned_codes.insert(code, true);
            num_banned += 1;
        }
    }

    std::cout << "Banned " << num_banned << " out of "
              << database.get_code_dictionary().size() << std::endl;

    // Compute the dictionary using map-reduce
    DictionaryData result = proccess_patients_in_parallel(
        database, 40,
        [&](DictionaryData& res, const Patient& p) {
            add_patient_to_dictionary(res, p, database.get_ontology(),
                                      database.size(), banned_codes);
        },
        merge_dictionary);

    // Regular dictionary entries
    std::vector<DictEntry> r_entries;

    // Hierarchical dictionary entries
    std::vector<DictEntry> h_entries;

    for (uint32_t code : result.code_counts.keys()) {
        double weight = *result.code_counts.find(code);
        DictEntry entry;
        entry.type = DictEntryType::CODE;
        entry.code_string = database.get_code_dictionary()[code];
        // Weight is the Shanon entropy
        entry.weight = weight * log(weight) + (1 - weight) * log(1 - weight);

        r_entries.push_back(entry);
    }
    for (uint32_t code : result.hierarchical_code_counts.keys()) {
        double weight = *result.hierarchical_code_counts.find(code);
        double baseline = 1;

        for (uint32_t parent : database.get_ontology().get_parents(code)) {
            baseline = std::min(*result.hierarchical_code_counts.find(parent),
                                baseline);
        }

        weight = weight / baseline;

        DictEntry entry;
        entry.type = DictEntryType::CODE;
        entry.code_string = database.get_code_dictionary()[code];
        // Make sure to use the hierarchical Shanon entropy formula
        entry.weight =
            baseline * (weight * log(weight) + (1 - weight) * log(1 - weight));

        h_entries.push_back(entry);
    }

    for (uint32_t code : result.text_counts.keys()) {
        auto* text_entries = result.text_counts.find(code);
        for (const auto& e : *text_entries) {
            DictEntry entry;
            entry.type = DictEntryType::TEXT;
            entry.code_string = database.get_code_dictionary()[code];
            entry.text_string = database.get_shared_text_dictionary()[e.first];
            entry.weight =
                e.second * log(e.second) + (1 - e.second) * log(1 - e.second);

            r_entries.push_back(entry);
            h_entries.push_back(entry);
        }
    }

    // Create percentile bins
    for (uint32_t code : result.numeric_samples.keys()) {
        auto* numeric = result.numeric_samples.find(code);

        std::vector<float> samples = numeric->get_samples();
        double weight = numeric->get_total_weight() /
                        10.0;  // Divide by 10 due to 10 bins by default

        std::sort(std::begin(samples), std::end(samples));

        size_t samples_per_bin = (samples.size() + 10) / 11;

        for (int bin = 0; bin < 10; bin++) {
            double start_val;
            if (bin == 0) {
                start_val = -std::numeric_limits<double>::max();
            } else {
                start_val = samples[bin * samples_per_bin];
            }

            double end_val;
            if (bin == 9) {
                end_val = std::numeric_limits<double>::max();
            } else {
                end_val = samples[(bin + 1) * samples_per_bin];
            }

            if (start_val == end_val) {
                continue;
            }

            DictEntry entry;
            entry.type = DictEntryType::NUMERIC;
            entry.code_string = database.get_code_dictionary()[code];
            entry.val_start = start_val;
            entry.val_end = end_val;
            entry.weight =
                weight * log(weight) + (1 - weight) * log(1 - weight);

            h_entries.push_back(entry);
            r_entries.push_back(entry);
        }
    }

    std::sort(std::begin(r_entries), std::end(r_entries));
    std::sort(std::begin(h_entries), std::end(h_entries));

    json age_stats;
    age_stats["mean"] = result.age_stats.get_mean();
    age_stats["std"] = result.age_stats.get_stddev();

    std::cout << "Got age statistics ... " << age_stats << std::endl;

    std::vector<float> hierarchical_counts(
        result.hierarchical_code_counts.size());

    for (uint32_t index : result.hierarchical_code_counts.keys()) {
        hierarchical_counts[index] =
            *result.hierarchical_code_counts.find(index);
    }

    json j;
    j["regular"] = r_entries;
    j["ontology_rollup"] = h_entries;
    j["age_stats"] = age_stats;
    j["hierarchical_counts"] = hierarchical_counts;

    std::vector<std::uint8_t> v = json::to_msgpack(j);

    std::ofstream o(output, std::ios_base::binary);

    o.write((const char*)v.data(), v.size());
}
