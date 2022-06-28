#include "patient2vec_extension.h"

#include <optional>
#include <random>

#include "blockingconcurrentqueue.h"
#include "civil_day_caster.h"
#include "flatmap.h"
#include "pybind11/pybind11.h"
#include "reader.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "numpy/numpy/ndarrayobject.h"
#include "picosha2.h"

namespace py = pybind11;

class OnlineStatistics {
   public:
    OnlineStatistics() {
        count = 0;

        current_mean = 0;
        variance = 0;
    }

    void add(double new_value) {
        count++;
        double delta = new_value - current_mean;
        current_mean += delta / count;
        double delta2 = new_value - current_mean;
        variance += delta * delta2;
        if (variance != variance || current_mean != current_mean ||
            variance < 0) {
            std::cout << "Got invalid variance or mean " << variance << " "
                      << current_mean << " " << new_value << std::endl;
            abort();
        }
    }

    double mean() const { return current_mean; }

    double standard_deviation() const { return sqrt(variance / (count - 1)); }

   private:
    uint64_t count;
    double current_mean;
    double variance;
};

template <typename T>
struct ReservoirSampler {
   public:
    ReservoirSampler(size_t _max_samples)
        : max_samples(_max_samples),
          num_samples_seen(0),
          next_sample(max_samples) {}

    void add(T value) {
        num_samples_seen++;

        if (num_samples_seen <= max_samples) {
            current_sample.push_back(std::move(value));
        } else if (next_sample == num_samples_seen) {
            std::uniform_int_distribution<> index_distribution(0,
                                                               max_samples - 1);
            size_t index_to_store = index_distribution(rng);
            current_sample[index_to_store] = std::move(value);
        }

        if (next_sample == num_samples_seen) {
            double prob_add = (double)max_samples / (1 + num_samples_seen);
            std::geometric_distribution<> gap_distribution(prob_add);
            next_sample += 1 + prob_add;
        }
    }

    absl::Span<const T> sample() const { return current_sample; }

   private:
    std::vector<T> current_sample;
    size_t max_samples;
    size_t num_samples_seen;
    size_t next_sample;
    std::mt19937_64 rng;
};

absl::flat_hash_set<std::string> bad_lab_values = {
    "not applicable",
    "see below:",
    "see below",
    "see report",
    "see note",
    "note:",
    "note",
    "null",
    "cancelled",
    "dnr",
    "dnrtnp",
    "qns",
    "nt",
    "nolav",
    "performed",
    "n/a",
    "sprcs",
    "prc",
};

std::string normalize_lab_value(std::string lab_value) {
    std::transform(lab_value.begin(), lab_value.end(), lab_value.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lab_value == "n" || lab_value == "neg" || lab_value == "neg." ||
        lab_value == "not detected" || lab_value == "none" ||
        lab_value == "none seen" || lab_value == "nosee" || lab_value == "nr" ||
        lab_value == "non reactive" || lab_value == "non-reactive") {
        lab_value = "negative";
    }

    if (lab_value == "p" || lab_value == "pos" || lab_value == "pos." ||
        lab_value == "reactive") {
        lab_value = "positive";
    }

    if (bad_lab_values.find(lab_value) != std::end(bad_lab_values)) {
        lab_value = "";
    }

    if (lab_value.find("tnp/", 0) == 0) {
        lab_value = "";
    }

    std::string_view lab_view = lab_value;

    if (lab_value.find("<", 0) == 0) {
        float value;
        bool converted = absl::SimpleAtof(lab_view.substr(1), &value);
        if (converted) {
            lab_value = absl::Substitute("<$0", value);
        }
    }

    return lab_value;
}

template <typename Iter>
int modulus(Iter first, Iter last, int mod) {
    int current_sum = 0;

    Iter current = first;

    while (current != last) {
        current_sum *= 256;
        current_sum += *current;

        current_sum %= mod;

        current++;
    }

    return current_sum;
}

bool determine_is_train(uint64_t seed, uint32_t patient_id) {
    std::string full_name = absl::Substitute("$0|$1", seed, patient_id);

    unsigned char hash[picosha2::k_digest_size];
    picosha2::hash256(std::begin(full_name), std::end(full_name),
                      std::begin(hash), std::end(hash));

    return modulus(std::begin(hash), std::end(hash), 10) != 9;
}

void get_all_candidates(uint32_t node, const OntologyReader& ontologies,
                        const FlatMap<uint32_t>& code_counts,
                        std::vector<std::pair<double, uint32_t>>& candidates,
                        uint32_t min_patient_count) {
    const uint32_t* parent_counts = code_counts.find(node);
    if (parent_counts == nullptr) {
        return;
    }
    for (uint32_t child : ontologies.get_children(node)) {
        const uint32_t* child_counts = code_counts.find(child);

        if (child_counts == nullptr || *child_counts < min_patient_count) {
            continue;
        }

        double entropy;
        if (*child_counts == *parent_counts) {
            entropy = 0;
            // std::cout<<"Bad code " << node << std::endl;
        } else {
            double ratio = (double)*child_counts / *parent_counts;

            double raw_entropy =
                ratio * std::log(ratio) + (1 - ratio) * std::log(1 - ratio);

            entropy = *parent_counts * raw_entropy;
        }

        candidates.push_back(std::make_pair(entropy, child));
        get_all_candidates(child, ontologies, code_counts, candidates,
                           min_patient_count);
    }
}

std::string create_info(const char* timeline_path, const char* ontology_path,
                        uint64_t seed, int min_patient_count) {
    ExtractReader reader(timeline_path, true);
    OntologyReader ontologies(ontology_path);

    auto iter = reader.iter();

    std::vector<std::pair<uint32_t, uint32_t>> train_patient_ids_with_length;
    std::vector<std::pair<uint32_t, uint32_t>> val_patient_ids_with_length;

    std::pair<OnlineStatistics, OnlineStatistics> age_stats;
    std::pair<OnlineStatistics, OnlineStatistics> delta_stats;

    auto add_stats = [](std::pair<OnlineStatistics, OnlineStatistics>& stats,
                        double value) {
        stats.first.add(value);
        stats.second.add(log(1 + value));
    };

    absl::flat_hash_set<uint32_t> recorded_date_codes(
        std::begin(ontologies.get_recorded_date_codes()),
        std::end(ontologies.get_recorded_date_codes()));

    FlatMap<uint32_t> code_counts;
    std::vector<uint32_t> patient_code_set;

    FlatMap<ReservoirSampler<float>> numeric_lab_samples;
    FlatMap<ReservoirSampler<uint32_t>> text_lab_samples;

    size_t num_patients = reader.get_patient_ids().size();

    size_t ten_percent = num_patients / 10;
    size_t sample_limit = num_patients / 1000;

    size_t num_processed = 0;

    for (auto patient_id : reader.get_patient_ids()) {
        bool is_train = determine_is_train(seed, patient_id);

        patient_code_set.clear();

        uint32_t length = 0;

        uint32_t last_age = 0;

        num_processed++;

        if (num_processed % ten_percent == 0) {
            std::cout << "Processed " << (100.0 * num_processed / num_patients)
                      << std::endl;
        }

        // if (num_processed % sample_limit == 0) {
        //     break;
        // }

        iter.process_patient(
            patient_id, [&](absl::CivilDay birth_day, uint32_t age,
                            const std::vector<uint32_t>& observations,
                            const std::vector<ObservationWithValue>&
                                observations_with_values) {
                length++;

                if (is_train) {
                    if (age != 0) {
                        add_stats(age_stats, age);

                        uint32_t delta = age - last_age;
                        if (last_age != 0) {
                            add_stats(delta_stats, delta);
                        }

                        last_age = age;
                    }

                    auto process_code = [&](uint32_t code) {
                        if (recorded_date_codes.find(code) ==
                            std::end(recorded_date_codes))
                            return;
                        for (const uint32_t& subword :
                             ontologies.get_subwords(code)) {
                            patient_code_set.push_back(subword);
                        }
                    };

                    for (const uint32_t& observation : observations) {
                        process_code(observation);
                    }

                    for (const auto& obs_with_value :
                         observations_with_values) {
                        process_code(obs_with_value.code);

                        if (obs_with_value.is_text) {
                            auto sampler = text_lab_samples.find_or_insert(
                                obs_with_value.code,
                                ReservoirSampler<uint32_t>(10000));
                            sampler->add(obs_with_value.text_value);
                        } else {
                            auto sampler = numeric_lab_samples.find_or_insert(
                                obs_with_value.code,
                                ReservoirSampler<float>(10000));
                            sampler->add(obs_with_value.numeric_value);
                        }
                    }
                }
            });

        if (length >= 3) {
            if (is_train) {
                std::sort(std::begin(patient_code_set),
                          std::end(patient_code_set));
                auto unique_end = std::unique(std::begin(patient_code_set),
                                              std::end(patient_code_set));

                for (auto iter = std::begin(patient_code_set);
                     iter != unique_end; iter++) {
                    uint32_t code = *iter;
                    auto* count = code_counts.find_or_insert(code, 0);
                    (*count)++;
                }

                train_patient_ids_with_length.push_back(
                    std::make_pair(patient_id, length));
            } else {
                val_patient_ids_with_length.push_back(
                    std::make_pair(patient_id, length));
            }
        }
    }

    uint32_t codes_with_no_siblings = 0;
    uint32_t codes_with_no_path_to_root = 0;

    uint32_t root_code = ontologies.get_root_code();

    std::map<std::string, uint32_t> final_code_counts;
    std::vector<std::pair<uint32_t, uint32_t>> valid_codes;
    for (uint32_t code = 0; code < code_counts.size(); code++) {
        uint32_t* count = code_counts.find(code);
        if (count == nullptr) {
            continue;
        }
        if (*count != 0) {
            final_code_counts[std::to_string(code)] = *count;
        }
        if (*count < (uint32_t)min_patient_count) {
            continue;
        }

        auto all_parents = ontologies.get_all_parents(code);
        bool has_root =
            std::find(std::begin(all_parents), std::end(all_parents),
                      root_code) != std::end(all_parents);

        if (has_root) {
            auto parents = ontologies.get_parents(code);

            bool has_sibling = false;

            for (auto parent : parents) {
                for (auto child : ontologies.get_children(parent)) {
                    uint32_t* child_counts = code_counts.find(child);
                    if (child != code && child < code_counts.size() &&
                        child_counts != nullptr &&
                        *child_counts >= (uint32_t)min_patient_count) {
                        has_sibling = true;
                    }
                }
            }

            if (!has_sibling) {
                codes_with_no_siblings++;
                continue;
            }
        } else {
            codes_with_no_path_to_root++;
        }

        valid_codes.push_back(std::make_pair(-*count, code));
    }

    std::cout << "Removed " << codes_with_no_siblings
              << " due to lack of siblings" << std::endl;
    std::cout << "Kept " << codes_with_no_path_to_root
              << " even with a lack of a path to the root" << std::endl;

    std::sort(std::begin(valid_codes), std::end(valid_codes));

    std::cout << "Got " << valid_codes.size() << " valid codes " << std::endl;

    std::map<std::string, uint32_t> valid_code_map;

    for (size_t i = 0; i < valid_codes.size(); i++) {
        uint32_t code = valid_codes[i].second;
        valid_code_map[std::to_string(code)] = i;
    }

    std::vector<std::pair<double, uint32_t>> candidates;
    get_all_candidates(*ontologies.get_dictionary().map("ICD10CM/ICD-10-CM"),
                       ontologies, code_counts, candidates, min_patient_count);

    std::sort(std::begin(candidates), std::end(candidates));

    std::map<std::string, uint32_t> valid_target_map;

    for (size_t i = 0; i < candidates.size(); i++) {
        // std::cout<<"Working with " << candidates[i].first << " " <<
        // candidates[i].second << std::endl;
        valid_target_map[std::to_string(candidates[i].second)] = i;
    }

    nlohmann::json result;

    result["code_counts"] = final_code_counts;
    result["valid_code_map"] = valid_code_map;
    result["valid_target_map"] = valid_target_map;
    result["train_patient_ids_with_length"] = train_patient_ids_with_length;
    result["val_patient_ids_with_length"] = val_patient_ids_with_length;

    std::vector<std::pair<uint32_t, std::vector<float>>> numeric_map;
    for (uint32_t i = 0; i < numeric_lab_samples.size(); i++) {
        auto* iter = numeric_lab_samples.find(i);
        if (iter == nullptr) {
            continue;
        }
        auto sample = iter->sample();
        if (sample.size() == 0) {
            continue;
        }
        numeric_map.push_back(std::make_pair(
            i, std::vector<float>(std::begin(sample), std::end(sample))));
    }

    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> text_map;
    for (uint32_t i = 0; i < text_lab_samples.size(); i++) {
        auto* iter = text_lab_samples.find(i);
        if (iter == nullptr) {
            continue;
        }
        auto sample = iter->sample();
        if (sample.size() == 0) {
            continue;
        }
        text_map.push_back(std::make_pair(
            i, std::vector<uint32_t>(std::begin(sample), std::end(sample))));
    }

    result["numeric_lab_map"] = numeric_map;
    result["text_lab_map"] = text_map;

    uint32_t next_lab_code = 0;

    nlohmann::json serialized;

    for (auto& item : numeric_map) {
        int num_splits = item.second.size() / 500;
        if (num_splits <= 1) {
            continue;
        }

        std::sort(std::begin(item.second), std::end(item.second));

        std::deque<float> numeric_ranges;
        std::vector<uint32_t> numeric_indices;

        float previous_value = std::numeric_limits<float>::quiet_NaN();

        for (int i = 0; i <= num_splits; i++) {
            float current_value;

            if (i == 0) {
                current_value = item.second[0];
            } else if (i == num_splits) {
                current_value = item.second[item.second.size() - 1];
            } else {
                current_value =
                    item.second[(item.second.size() * i) / num_splits];
            }

            if (current_value != previous_value) {
                numeric_indices.push_back(next_lab_code++);
                numeric_ranges.push_back(current_value);
            }
            previous_value = current_value;
        }

        if (numeric_ranges.size() > 2) {
            numeric_ranges.pop_back();
            numeric_ranges.pop_front();
            numeric_indices.pop_back();

            next_lab_code--;

            serialized[std::to_string(item.first)]["numeric_ranges"] =
                numeric_ranges;
            serialized[std::to_string(item.first)]["numeric_indices"] =
                numeric_indices;
        } else {
            next_lab_code -= numeric_indices.size();
        }
    }

    for (auto& item : text_map) {
        std::map<std::string, std::set<uint32_t>> mapping;
        std::map<std::string, int> count_map;

        for (uint32_t text_value : item.second) {
            std::string normalized = normalize_lab_value(std::string(
                *reader.get_value_dictionary().get_word(text_value)));
            if (normalized == "") {
                continue;
            }
            mapping[normalized].insert(text_value);
            count_map[normalized] += 1;
        }

        uint32_t valid_found = 0;

        for (auto& iter : count_map) {
            if (iter.second > 100) {
                valid_found++;
            }
        }

        if (valid_found <= 1) {
            continue;
        }

        for (auto& iter : count_map) {
            if (iter.second > 100) {
                uint32_t index = next_lab_code++;
                for (uint32_t token : mapping[iter.first]) {
                    serialized[std::to_string(item.first)]["text_indices"]
                              [std::to_string(token)] = index;
                    serialized[std::to_string(item.first)]["text_values"]
                              [std::to_string(token)] = iter.first;
                }
            }
        }
    }

    result["lab_value_map"] = serialized;
    result["num_lab_codes"] = next_lab_code;

    std::cout << "Got " << next_lab_code << " lab codes " << std::endl;

    auto extract_combined_stat =
        [&result](std::pair<OnlineStatistics, OnlineStatistics>& stats,
                  const std::string& name) {
            result[name + "_mean"] = stats.first.mean();
            result[name + "_std"] = stats.first.standard_deviation();

            result[name + "_log_mean"] = stats.second.mean();
            result[name + "_log_std"] = stats.second.standard_deviation();
        };

    extract_combined_stat(age_stats, "age");
    extract_combined_stat(delta_stats, "delta");

    return result.dump();
}

class LabData {
   public:
    LabData(uint32_t index_offset, const nlohmann::json& json_data) {
        auto number_ptr = json_data.find("numeric_indices");

        if (number_ptr != std::end(json_data)) {
            for (const auto& value : *number_ptr) {
                uint32_t correct_value = (uint32_t)value + index_offset;
                numeric_indices.push_back(correct_value);
                all_codes.push_back(correct_value);
            }

            for (const auto& value : json_data["numeric_ranges"]) {
                numeric_dividers.push_back((float)value);
            }
        }

        auto string_ptr = json_data.find("text_indices");

        if (string_ptr != std::end(json_data)) {
            for (const auto& entry : string_ptr->items()) {
                uint32_t key = std::stoi(entry.key());
                uint32_t value = ((uint32_t)entry.value()) + index_offset;

                string_values.insert(std::make_pair(key, value));
                all_codes.push_back(value);
            }
        }
    }

    const std::vector<uint32_t>& get_all_codes() const { return all_codes; }

    std::optional<uint32_t> get_index(float value) const {
        if (numeric_dividers.size() == 0) {
            return std::nullopt;
        }

        size_t index;

        for (index = 0; index < numeric_dividers.size(); index++) {
            if (value <= numeric_dividers[index]) {
                break;
            }
        }

        return numeric_indices[index];
    }

    std::optional<uint32_t> get_index(uint32_t string_value) const {
        auto iter = string_values.find(string_value);
        if (iter == std::end(string_values)) {
            return std::nullopt;
        } else {
            return {iter->second};
        }
    }

   private:
    std::vector<uint32_t> all_codes;
    absl::flat_hash_map<uint32_t, uint32_t> string_values;
    std::vector<uint32_t> numeric_indices;
    std::vector<float> numeric_dividers;
};

std::vector<int64_t> convert_to_int(py::object data) {
    py::object int_data = py::reinterpret_steal<py::object>(
        (PyObject*)PyArray_FromAny(data.ptr(), PyArray_DescrFromType(NPY_INT64),
                                   1, 1, NPY_ARRAY_CARRAY_RO, nullptr));
    const int64_t* ptr =
        (const int64_t*)PyArray_DATA((PyArrayObject*)int_data.ptr());
    size_t size = PyArray_SHAPE((PyArrayObject*)int_data.ptr())[0];
    std::vector<int64_t> result(ptr, ptr + size);
    return result;
}

std::vector<std::pair<uint32_t, uint32_t>> create_lengths(
    const std::map<uint32_t, std::vector<std::pair<uint32_t, bool>>>&
        label_map) {
    std::vector<std::pair<uint32_t, uint32_t>> result;

    for (const auto& item : label_map) {
        uint32_t patient_id = item.first;
        uint32_t max_index = item.second.back().first;
        if (max_index <= 0) {
            continue;
        }
        result.push_back(std::make_pair(patient_id, max_index + 1));
    }

    return result;
}

std::map<uint32_t, std::vector<std::pair<uint32_t, bool>>> create_label_map(
    py::tuple data) {
    auto results = convert_to_int(data[0]);
    auto patient_ids = convert_to_int(data[1]);
    auto patient_day_indices = convert_to_int(data[2]);

    std::map<uint32_t, std::vector<std::pair<uint32_t, bool>>> label_map;

    for (size_t i = 0; i < results.size(); i++) {
        uint32_t patient_id = patient_ids[i];
        label_map[patient_id].push_back(
            std::make_pair(patient_day_indices[i] + 1, (bool)results[i]));
    }

    for (auto& iter : label_map) {
        std::sort(std::begin(iter.second), std::end(iter.second));
    }

    return label_map;
}

class PTDatasetIterator;

class PatientTimelineDataset {
   public:
    friend PTDatasetIterator;

    PatientTimelineDataset(const char* compressed_extract_path,
                           const char* ontology_path, const char* info_path,
                           py::tuple train_data, py::tuple val_data)
        : PatientTimelineDataset(compressed_extract_path, ontology_path,
                                 info_path) {
        train_map = create_label_map(train_data);
        train_lengths = create_lengths(train_map);

        val_map = create_label_map(val_data);
        val_lengths = create_lengths(val_map);

        std::stable_sort(std::begin(train_lengths), std::end(train_lengths),
                         [](const auto& first, const auto& second) {
                             return first.second > second.second;
                         });
    }

    PatientTimelineDataset(const char* timelines_path,
                           const char* ontology_path, const char* info_path)
        : timelines(timelines_path, true), ontologies(ontology_path) {
        {
            for (uint32_t code : ontologies.get_recorded_date_codes()) {
                valid_recorded_date_code.insert(code, true);
            }

            nlohmann::json info;
            std::ifstream info_file(info_path);
            info_file >> info;

            threshold = info["threshold"];
            num_lab_codes = info["num_lab_codes"];

            age_mean = info["age_mean"];
            age_std = info["age_std"];
            log_age_mean = info["age_log_mean"];
            log_age_std = info["age_log_std"];

            delta_mean = info["delta_mean"];
            delta_std = info["delta_std"];
            log_delta_mean = info["delta_log_mean"];
            log_delta_std = info["delta_log_std"];

            for (const auto& value : info["train_patient_ids_with_length"]) {
                train_lengths.push_back(std::make_pair(value[0], value[1]));
            }

            for (const auto& value : info["val_patient_ids_with_length"]) {
                val_lengths.push_back(std::make_pair(value[0], value[1]));
            }

            for (const auto& entry : info["valid_code_map"].items()) {
                uint32_t key = std::stoi(entry.key());
                uint32_t value = entry.value();

                valid_code_map.insert(key, value);
            }

            num_valid_codes = info["valid_code_map"].size();

            for (const auto& entry : info["lab_value_map"].items()) {
                uint32_t key = std::stoi(entry.key());
                LabData value(threshold, entry.value());
                lab_data_map.insert(key, std::move(value));
            }

            num_valid_targets = 5001;

            for (const auto& entry : info["valid_target_map"].items()) {
                uint32_t key = std::stoi(entry.key());
                uint32_t value = entry.value();

                if (value < num_valid_targets - 1) {
                    valid_target_map.insert(key, value);
                }
            }
        }

        {
            for (int i = 0; i < 50; i++) {
                double pow = (i - 10) / 39.0;
                rates[i] = 1.0 / std::pow(1000, pow);
            }
        }

        std::stable_sort(std::begin(train_lengths), std::end(train_lengths),
                         [](const auto& first, const auto& second) {
                             return first.second > second.second;
                         });

        int current_sum = 0;
        int total = 210;
        for (int i = 0; i < 20; i++) {
            starts[i] = (float)current_sum / total * 365 * 20;
            current_sum += (i + 1);
            ends[i] = (float)current_sum / total * 365 * 20;
        }

        icd_root = ontologies.get_dictionary().map("ICD10CM/ICD-10-CM").value();
        ontology_dictionary = ontologies.get_dictionary();
    }

    std::unique_ptr<PTDatasetIterator> get_iterator(
        bool is_val, int batch_size, uint64_t seed, float day_dropout = 0,
        float code_dropout = 0) const {
        return std::make_unique<PTDatasetIterator>(
            *this, is_val, batch_size, seed, day_dropout, code_dropout);
    }

    int num_batches(int batch_size, bool is_val = false) const {
        std::vector<std::pair<uint32_t, uint32_t>> lengths_vec;

        int result = 0;

        size_t current_index = 0;

        lengths_vec = is_val ? val_lengths : train_lengths;

        while (current_index < lengths_vec.size()) {
            int current_length = lengths_vec[current_index].second;
            int num_items = std::max(1, batch_size / current_length);

            current_index += num_items;

            if (current_length <= 512) {
                result++;
            }
        }

        return result;
    }

   private:
    uint32_t num_valid_codes;
    uint32_t num_lab_codes;
    uint32_t num_valid_targets;
    uint32_t threshold;

    std::optional<uint32_t> get_valid_code(uint32_t code) const {
        if (code == ontologies.get_root_code()) {
            return std::nullopt;
        } else {
            auto* ptr = valid_code_map.find(code);
            if (ptr == nullptr) {
                return std::nullopt;
            } else {
                return *ptr;
            }
        }
    }

    bool is_valid_recorded_date_code(uint32_t code) const {
        return valid_recorded_date_code.find(code) != nullptr;
    }

    ExtractReader timelines;
    OntologyReader ontologies;

    FlatMap<uint32_t> valid_target_map;
    FlatMap<uint32_t> valid_code_map;
    FlatMap<bool> valid_recorded_date_code;
    FlatMap<LabData> lab_data_map;

    std::vector<std::pair<uint32_t, uint32_t>> train_lengths;
    std::vector<std::pair<uint32_t, uint32_t>> val_lengths;

    std::array<float, 50> rates;

    std::map<uint32_t, std::vector<std::pair<uint32_t, bool>>> train_map;
    std::map<uint32_t, std::vector<std::pair<uint32_t, bool>>> val_map;

    double age_mean, age_std, log_age_mean, log_age_std;
    double delta_mean, delta_std, log_delta_mean, log_delta_std;

    absl::flat_hash_set<uint32_t> bad_to_predict;

    uint32_t icd_root;

    std::array<float, 20> ends;
    std::array<float, 20> starts;

    TermDictionary ontology_dictionary;
};

void get_negative_codes(const OntologyReader& ontologies,
                        const std::vector<uint32_t>& positive_codes,
                        std::vector<uint32_t>& negative_codes) {
    for (uint32_t node : positive_codes) {
        for (uint32_t child : ontologies.get_children(node)) {
            if (std::find(std::begin(positive_codes), std::end(positive_codes),
                          child) == std::end(positive_codes)) {
                negative_codes.push_back(child);
            }
        }
    }
}

class PTDatasetBatch {
   public:
    std::unique_ptr<ExtractReaderIterator> iter;

    size_t index;

    std::vector<uint32_t> positive_codes_per_day;
    std::vector<bool> drop_flags;

    std::vector<uint32_t> lab_codes;

    std::vector<uint32_t> buffer;

    std::vector<int64_t> patient_ids;
    std::vector<int64_t> codes;
    std::vector<int64_t> offsets;
    std::vector<int64_t> codes1;
    std::vector<int64_t> offsets1;

    std::vector<int64_t> label_indices;
    std::vector<float> labels;

    std::vector<std::array<int64_t, 2>> lengths;
    std::vector<std::array<float, 5>> day_info;

    std::vector<std::array<float, 200>> pos_encoding;

    FlatMap<uint32_t> target_ages;

    std::vector<std::array<uint64_t, 2>> target_indices;
    std::vector<float> target_labels;
    std::vector<float> target_factors;
    std::vector<std::array<uint64_t, 2>> target_other_indices;

    std::vector<std::array<uint64_t, 2>> target_nonfactor_indices;
    std::vector<float> target_nonfactor_labels;
    std::vector<std::array<uint64_t, 2>> target_nonfactor_other_indices;

    std::vector<int64_t> day_index;
};

class PTDatasetIterator {
   public:
    PTDatasetIterator(const PatientTimelineDataset& p, bool is_val_,
                      int batch_size, uint64_t seed, float day_dropout_ = 0,
                      float code_dropout_ = 0)
        : rng(seed),
          parent(p),
          is_val(is_val_),
          day_dropout(day_dropout_),
          code_dropout(code_dropout_) {
        if (is_val) {
            patient_lengths = parent.val_lengths;
            label_map = parent.val_map;
        } else {
            patient_lengths = parent.train_lengths;
            label_map = parent.train_map;
        }

        std::shuffle(std::begin(patient_lengths), std::end(patient_lengths),
                     rng);

        std::stable_sort(std::begin(patient_lengths), std::end(patient_lengths),
                         [](const auto& first, const auto& second) {
                             return first.second > second.second;
                         });

        size_t current_index = 0;

        while (current_index < patient_lengths.size()) {
            int current_length = patient_lengths[current_index].second;

            int num_items = std::max(1, batch_size / current_length);

            int end_index =
                std::min(current_index + num_items, patient_lengths.size());

            if (current_length <= 512) {
                indices.push_back(std::make_pair(current_index, end_index));
            }

            current_index = end_index;
        }

        std::shuffle(std::begin(indices), std::end(indices), rng);

        next_index = 0;

        num_threads = 4;
        done_threads = 0;

        batches.resize(num_threads * 2 + 1);

        for (size_t i = 0; i < num_threads * 2 + 1 && i < indices.size(); i++) {
            PTDatasetBatch* batch = &batches[i];
            batch->iter =
                std::make_unique<ExtractReaderIterator>(p.timelines.iter());
            batch->index = next_index++;
            add_batch_to_empty(batch);
        }

        for (size_t i = 0; i < num_threads; i++) {
            uint64_t seed_for_thread = rng();
            PTDatasetIterator* self = this;
            std::thread thread([self, i, seed_for_thread]() {
                std::mt19937_64 batch_rng(seed_for_thread);
                while (true) {
                    PTDatasetBatch* batch = self->get_next_empty_batch();
                    if (batch == nullptr) {
                        self->add_batch_to_result(nullptr);
                        break;
                    } else {
                        self->fill_in_batch(batch_rng, *batch);
                        self->add_batch_to_result(batch);
                    }
                }
            });

            threads.push_back(std::move(thread));
        }

        current_batch = nullptr;
    }

    ~PTDatasetIterator() {
        for (size_t i = 0; i < num_threads; i++) {
            add_batch_to_empty(nullptr);
        }

        for (size_t i = 0; i < num_threads; i++) {
            threads[i].join();
        }
    }

    void special_targets(PTDatasetBatch& batch, uint32_t patient_id,
                         uint64_t patient_index, uint32_t node, uint32_t offset,
                         uint32_t current_age, uint32_t censor_age) const {
        const uint32_t* target_index = parent.valid_target_map.find(node);

        auto term = parent.ontology_dictionary.get_word(node).value_or("");

        const uint32_t* age = batch.target_ages.find(node);

        bool is_censor;
        uint32_t target_age;

        if (age == nullptr) {
            is_censor = true;
            target_age = censor_age;
        } else {
            is_censor = false;
            target_age = *age;

            for (uint32_t child : parent.ontologies.get_children(node)) {
                special_targets(batch, patient_id, patient_index, child, offset,
                                target_age, censor_age);
            }
        }

        if (target_index != nullptr) {
            // if (target_index != nullptr && term == "ICD10CM/E00-E89") {
            for (size_t bin_index = 0; bin_index < parent.ends.size();
                 bin_index++) {
                float start = parent.starts[bin_index] + offset;
                float end = parent.ends[bin_index] + offset;

                bool is_start = start <= current_age && current_age < end;
                bool is_final = start <= target_age && target_age < end;
                bool is_valid = start < target_age && current_age < end;

                if (!is_valid) {
                    continue;
                }

                uint64_t index = *target_index * 2 * 20 +
                                 (is_start ? 0 : 1) * 20 + bin_index;

                if (is_final) {
                    if (is_censor) {
                        float factor =
                            (target_age - start) / ((float)(end - start));
                        batch.target_indices.push_back({patient_index, index});
                        batch.target_other_indices.push_back(
                            {patient_index, *target_index});
                        batch.target_labels.push_back(0);
                        batch.target_factors.push_back(factor);
                    } else {
                        batch.target_nonfactor_indices.push_back(
                            {patient_index, index});
                        batch.target_nonfactor_other_indices.push_back(
                            {patient_index, *target_index});
                        batch.target_nonfactor_labels.push_back(1);
                    }
                } else {
                    batch.target_nonfactor_indices.push_back(
                        {patient_index, index});
                    batch.target_nonfactor_other_indices.push_back(
                        {patient_index, *target_index});
                    batch.target_nonfactor_labels.push_back(0);
                }
            }
        }
    }

    void fill_in_batch(std::mt19937_64& batch_rng,
                       PTDatasetBatch& batch) const {
        // printf("Starting %d\n", (int) batch.index);
        int start_index, end_index;

        std::tie(start_index, end_index) = indices.at(batch.index);

        batch.codes.clear();
        batch.offsets.clear();
        batch.lengths.clear();

        batch.codes1.clear();
        batch.offsets1.clear();

        batch.day_info.clear();
        batch.patient_ids.clear();

        batch.pos_encoding.clear();

        // batch.target_times.clear();
        // batch.censor_masks.clear();
        // batch.event_masks.clear();

        batch.target_other_indices.clear();
        batch.target_indices.clear();
        batch.target_labels.clear();
        batch.target_factors.clear();

        batch.target_nonfactor_labels.clear();
        batch.target_nonfactor_indices.clear();
        batch.target_nonfactor_other_indices.clear();

        batch.label_indices.clear();
        batch.labels.clear();

        batch.day_index.clear();

        uint32_t max_days_to_drop = 0;

        if (patient_lengths.at(start_index).second > 3) {
            max_days_to_drop = patient_lengths.at(start_index).second - 3;
        }

        std::binomial_distribution<> day_dropout_distribution(max_days_to_drop,
                                                              day_dropout);
        std::bernoulli_distribution code_dropout_distribution(code_dropout);

        int days_to_drop = day_dropout_distribution(batch_rng);

        int max_length =
            patient_lengths.at(start_index).second - 1 - days_to_drop;

        batch.day_index.resize(max_length * (end_index - start_index), -1);

        // size_t num_target_entries = max_length * (end_index - start_index) *
        // parent.num_valid_targets;
        // batch.target_times.resize(num_target_entries, 1);
        // batch.censor_masks.resize(num_target_entries, 0);
        // batch.event_masks.resize(num_target_entries, 0);

        for (int i = 0; i < (end_index - start_index); i++) {
            uint32_t patient_id = patient_lengths.at(start_index + i).first;

            absl::Span<const std::pair<uint32_t, bool>> labels;

            if (!label_map.empty()) {
                auto iter = label_map.find(patient_id);
                if (iter == std::end(label_map)) {
                    std::cout << absl::Substitute(
                        "Missing labels for patient id ? $0\n", patient_id);
                    abort();
                }

                labels = iter->second;
            }
            // size_t next_label_index = 0;

            batch.patient_ids.push_back(patient_id);

            uint32_t length_delta =
                (patient_lengths.at(start_index).second -
                 patient_lengths.at(start_index + i).second);

            uint32_t days_to_drop_for_patient = days_to_drop;

            if (length_delta >= days_to_drop_for_patient) {
                days_to_drop_for_patient = 0;
            } else {
                days_to_drop_for_patient -= length_delta;
            }

            uint32_t length = patient_lengths.at(start_index + i).second - 1;

            if (length <= 1) {
                printf("Bad Bad %d %d \n", length, patient_id);
            }

            uint32_t length_with_drop = length - days_to_drop_for_patient;

            batch.drop_flags.clear();

            for (size_t flag_i = 0; flag_i < length; flag_i++) {
                batch.drop_flags.push_back(flag_i < days_to_drop_for_patient);
            }

            std::shuffle(std::begin(batch.drop_flags),
                         std::end(batch.drop_flags), batch_rng);

            batch.lengths.push_back(
                {(int64_t)batch.offsets.size(), (int64_t)(length_with_drop)});

            int32_t current_offset = 0;
            uint32_t last_age = 0;
            int64_t day_index = 0;

            batch.target_ages.clear();

            uint32_t censor_age = 0;

            bool found = batch.iter->process_patient(
                patient_id, [&](absl::CivilDay birth_day, uint32_t age,
                                const std::vector<uint32_t>& observations,
                                const std::vector<ObservationWithValue>&
                                    observations_with_values) {
                    censor_age = age;

                    for (uint32_t code : observations) {
                        if (!parent.is_valid_recorded_date_code(code)) {
                            continue;
                        }

                        for (uint32_t subword :
                             parent.ontologies.get_subwords(code)) {
                            batch.target_ages.find_or_insert(subword, age);
                        }
                    }
                });

            if (!found) {
                std::cout << "Could not find patient_id " << patient_id
                          << std::endl;
                abort();
            }

            found = batch.iter->process_patient(
                patient_id, [&](absl::CivilDay birth_day, uint32_t age,
                                const std::vector<uint32_t>& observations,
                                const std::vector<ObservationWithValue>&
                                    observations_with_values) {
                    bool is_drop;

                    if (day_index > length) {
                        return;
                    } else if (day_index == length) {
                        is_drop = false;
                    } else {
                        is_drop = batch.drop_flags[day_index];
                    }

                    batch.positive_codes_per_day.clear();
                    batch.lab_codes.clear();

                    for (uint32_t code : observations) {
                        if (!parent.is_valid_recorded_date_code(code)) {
                            continue;
                        }

                        if (code_dropout != 0 &&
                            code_dropout_distribution(batch_rng)) {
                            continue;
                        }

                        for (uint32_t subword :
                             parent.ontologies.get_subwords(code)) {
                            batch.positive_codes_per_day.push_back(subword);
                        }
                    }

                    for (auto code_with_value : observations_with_values) {
                        uint32_t code = code_with_value.code;

                        if (!parent.is_valid_recorded_date_code(code)) {
                            continue;
                        }

                        if (code_dropout != 0 &&
                            code_dropout_distribution(batch_rng)) {
                            continue;
                        }

                        for (uint32_t subword :
                             parent.ontologies.get_subwords(code)) {
                            batch.positive_codes_per_day.push_back(subword);
                        }

                        auto lab_data = parent.lab_data_map.find(code);
                        if (lab_data != nullptr) {
                            std::optional<uint32_t> value_token;
                            if (code_with_value.is_text) {
                                value_token = lab_data->get_index(
                                    code_with_value.text_value);
                            } else {
                                value_token = lab_data->get_index(
                                    code_with_value.numeric_value);
                            }

                            if (value_token) {
                                batch.lab_codes.push_back(*value_token);
                            }
                        }
                    }

                    // Deal with normal tokens

                    std::sort(std::begin(batch.positive_codes_per_day),
                              std::end(batch.positive_codes_per_day));
                    auto last =
                        std::unique(std::begin(batch.positive_codes_per_day),
                                    std::end(batch.positive_codes_per_day));
                    batch.positive_codes_per_day.erase(
                        last, std::end(batch.positive_codes_per_day));

                    // Deal with lab value tokens

                    std::sort(std::begin(batch.lab_codes),
                              std::end(batch.lab_codes));
                    last = std::unique(std::begin(batch.lab_codes),
                                       std::end(batch.lab_codes));
                    batch.lab_codes.erase(last, std::end(batch.lab_codes));

                    // bool has_label = false;
                    // bool label = false;

                    // while (next_label_index < labels.size()) {
                    //     uint32_t next_label_day;
                    //     bool next_label_value;
                    //     std::tie(next_label_day, next_label_value) =
                    //     labels[next_label_index];

                    //     if (next_label_day < day_index) {
                    //         next_label_index++;
                    //     } else if (next_label_day == day_index) {
                    //         has_label = true;
                    //         label = next_label_value;
                    //         break;
                    //     } else if (next_label_day > day_index) {
                    //         break;
                    //     }
                    // }

                    if (day_index != length && !is_drop) {
                        // Add to feature
                        batch.day_index.at(i * max_length + current_offset) =
                            day_index;

                        if (age == 0) {
                            batch.day_info.push_back({
                                0,
                                0,
                                0,
                                0,
                                1,
                            });
                        } else {
                            double age_double = age;
                            double log_age = std::log(1 + age_double);

                            if (last_age != 0) {
                                double delta_double = (age - last_age);
                                double log_delta = std::log(1 + delta_double);
                                batch.day_info.push_back({
                                    (float)((age_double - parent.age_mean) /
                                            parent.age_std),
                                    (float)((log_age - parent.log_age_mean) /
                                            parent.log_age_std),
                                    (float)((delta_double - parent.delta_mean) /
                                            parent.delta_std),
                                    (float)((log_delta -
                                             parent.log_delta_mean) /
                                            parent.log_delta_std),
                                    0,
                                });
                            } else {
                                batch.day_info.push_back({
                                    (float)((age_double - parent.age_mean) /
                                            parent.age_std),
                                    (float)((log_age - parent.log_age_mean) /
                                            parent.log_age_std),
                                    0,
                                    0,
                                    0,
                                });
                            }
                        }

                        std::array<float, 200> position;

                        for (int pos_i = 0; pos_i < 50; pos_i++) {
                            position[pos_i * 4 + 0] =
                                std::sin(current_offset * parent.rates[pos_i]);
                            position[pos_i * 4 + 1] =
                                std::cos(current_offset * parent.rates[pos_i]);
                            position[pos_i * 4 + 2] = std::sin(
                                ((float)age / 365) * parent.rates[pos_i]);
                            position[pos_i * 4 + 3] = std::cos(
                                ((float)age / 365) * parent.rates[pos_i]);
                        }

                        batch.pos_encoding.push_back(position);

                        batch.offsets.push_back(batch.codes.size());
                        batch.offsets1.push_back(batch.codes1.size());

                        bool added_one = false;
                        bool added_one1 = false;

                        for (uint32_t code : batch.positive_codes_per_day) {
                            if (auto valid_code = parent.get_valid_code(code)) {
                                if (*valid_code >= parent.threshold) {
                                    batch.codes1.push_back(*valid_code -
                                                           parent.threshold);
                                    added_one1 = true;
                                } else {
                                    batch.codes.push_back(*valid_code);
                                    added_one = true;
                                }
                            }
                        }

                        for (uint32_t code : batch.lab_codes) {
                            if (code_dropout == 0 ||
                                !code_dropout_distribution(batch_rng)) {
                                batch.codes.push_back(code);
                                added_one = true;
                            }
                        }

                        if (!added_one) {
                            batch.codes.push_back(parent.threshold +
                                                  parent.num_lab_codes);
                        }

                        if (!added_one1) {
                            batch.codes1.push_back(parent.num_valid_codes -
                                                   parent.threshold);
                        }

                        last_age = age;

                        // Add labels

                        if (age != 0) {
                            uint64_t patient_index =
                                i * max_length + current_offset;
                            special_targets(batch, patient_id, patient_index,
                                            parent.icd_root, age, age,
                                            censor_age);

                            // for (uint32_t target = 0; target <
                            // parent.num_valid_targets; target++) {

                            //     float time;
                            //     bool censor_mask;
                            //     bool event_mask;

                            //     if (target == parent.num_valid_targets - 1) {
                            //         // The special censoring target
                            //         time = (censor_age - age);
                            //         censor_mask = false;
                            //         event_mask = true;
                            //     } else {
                            //         if (batch.target_ages[target] == 0) {
                            //             // Censored
                            //             time = (censor_age - age);
                            //             censor_mask = true;
                            //             event_mask = false;
                            //         } else if (batch.target_ages[target] <=
                            //         age) {
                            //             // Already happened
                            //             time = 1;
                            //             censor_mask = false;
                            //             event_mask = false;
                            //         } else if (batch.target_ages[target] >
                            //         age) {
                            //             // Will happen in the future
                            //             time = batch.target_ages[target] -
                            //             age; censor_mask = false; event_mask
                            //             = true;
                            //         }
                            //     }

                            //     size_t index = max_length *
                            //     parent.num_valid_targets * i
                            //     + parent.num_valid_targets * current_offset +
                            //     target;

                            //     batch.target_times[index] = time;
                            //     batch.censor_masks[index] = censor_mask;
                            //     batch.event_masks[index] = event_mask;
                            // }
                        }

                        current_offset++;
                    }

                    day_index++;
                });

            if (day_index != length + 1) {
                std::cout << "Day index should count up to length + 1? "
                          << day_index << " " << length << " " << patient_id
                          << " " << patient_lengths.at(start_index + i).second
                          << std::endl;
                abort();
            }

            // ? 33 31 2671567

            if (!found) {
                std::cout << "Could not find patient_id " << patient_id
                          << std::endl;
                abort();
            }
        }

        batch.target_indices.insert(std::end(batch.target_indices),
                                    std::begin(batch.target_nonfactor_indices),
                                    std::end(batch.target_nonfactor_indices));
        batch.target_labels.insert(std::end(batch.target_labels),
                                   std::begin(batch.target_nonfactor_labels),
                                   std::end(batch.target_nonfactor_labels));

        batch.target_other_indices.insert(
            std::end(batch.target_other_indices),
            std::begin(batch.target_nonfactor_other_indices),
            std::end(batch.target_nonfactor_other_indices));
    }

    py::dict next() {
        {
            py::gil_scoped_release release;

            if (current_batch != nullptr) {
                if (next_index == indices.size()) {
                    // Done
                    for (size_t i = 0; i < num_threads; i++) {
                        add_batch_to_empty(nullptr);
                    }
                    next_index++;
                } else if (next_index < indices.size()) {
                    current_batch->index = next_index++;
                    add_batch_to_empty(current_batch);
                }

                current_batch = nullptr;
            }

            while (current_batch == nullptr) {
                current_batch = get_next_result_batch();

                if (current_batch == nullptr) {
                    done_threads++;

                    if (done_threads == num_threads) {
                        break;
                    }
                }
            }
        }

        if (current_batch == nullptr) {
            throw py::stop_iteration();
        }

        py::dict result;

        npy_intp patient_id_numpy_dims[] = {
            (npy_intp)current_batch->patient_ids.size()};
        py::handle patient_id_numpy =
            PyArray_SimpleNewFromData(1, patient_id_numpy_dims, NPY_INT64,
                                      current_batch->patient_ids.data());

        result["pid"] = patient_id_numpy;

        npy_intp day_index_numpy_dims[] = {
            (npy_intp)current_batch->patient_ids.size(),
            (npy_intp)(current_batch->day_index.size() /
                       current_batch->patient_ids.size())};
        py::handle day_index_numpy =
            PyArray_SimpleNewFromData(2, day_index_numpy_dims, NPY_INT64,
                                      current_batch->day_index.data());

        result["day_index"] = day_index_numpy;

        npy_intp codes_numpy_dims[] = {(npy_intp)current_batch->codes.size()};
        py::handle codes_numpy = PyArray_SimpleNewFromData(
            1, codes_numpy_dims, NPY_INT64, current_batch->codes.data());

        npy_intp offsets_numpy_dims[] = {
            (npy_intp)current_batch->offsets.size()};
        py::handle offsets_numpy = PyArray_SimpleNewFromData(
            1, offsets_numpy_dims, NPY_INT64, current_batch->offsets.data());

        npy_intp codes1_numpy_dims[] = {(npy_intp)current_batch->codes1.size()};
        py::handle codes1_numpy = PyArray_SimpleNewFromData(
            1, codes1_numpy_dims, NPY_INT64, current_batch->codes1.data());

        npy_intp offsets1_numpy_dims[] = {
            (npy_intp)current_batch->offsets1.size()};
        py::handle offsets1_numpy = PyArray_SimpleNewFromData(
            1, offsets1_numpy_dims, NPY_INT64, current_batch->offsets1.data());

        npy_intp length_numpy_dims[] = {(npy_intp)current_batch->lengths.size(),
                                        2};
        py::handle lengths_numpy = PyArray_SimpleNewFromData(
            2, length_numpy_dims, NPY_INT64, current_batch->lengths.data());

        npy_intp pos_encoding_numpy_dims[] = {
            (npy_intp)current_batch->pos_encoding.size(), 200};
        py::handle pos_encoding_numpy =
            PyArray_SimpleNewFromData(2, pos_encoding_numpy_dims, NPY_FLOAT32,
                                      current_batch->pos_encoding.data());

        npy_intp day_info_numpy_dims[] = {
            (npy_intp)current_batch->day_info.size(), 5};
        py::handle day_info_numpy =
            PyArray_SimpleNewFromData(2, day_info_numpy_dims, NPY_FLOAT32,
                                      current_batch->day_info.data());

        result["rnn"] = py::make_tuple(codes_numpy, offsets_numpy, codes1_numpy,
                                       offsets1_numpy, day_info_numpy,
                                       pos_encoding_numpy, lengths_numpy);

        if (label_map.empty()) {
            // npy_intp target_dims[] = {
            //     (npy_intp) current_batch->patient_ids.size(),
            //     (npy_intp) (current_batch->day_index.size() /
            //     current_batch->patient_ids.size()), parent.num_valid_targets,
            // };

            // py::handle target_times_numpy = PyArray_SimpleNewFromData(3,
            // target_dims, NPY_FLOAT32, current_batch->target_times.data());
            // py::handle target_censor_numpy = PyArray_SimpleNewFromData(3,
            // target_dims, NPY_BOOL, current_batch->censor_masks.data());
            // py::handle target_event_numpy = PyArray_SimpleNewFromData(3,
            // target_dims, NPY_BOOL, current_batch->event_masks.data());

            // result["task"] = py::make_tuple(target_times_numpy,
            // target_censor_numpy, target_event_numpy);

            npy_intp target_indices_dims[] = {
                (npy_intp)current_batch->target_indices.size(),
                2,
            };

            py::handle target_indices_numpy =
                PyArray_SimpleNewFromData(2, target_indices_dims, NPY_INT64,
                                          current_batch->target_indices.data());

            npy_intp target_labels_dims[] = {
                (npy_intp)current_batch->target_labels.size(),
            };

            py::handle target_labels_numpy =
                PyArray_SimpleNewFromData(1, target_labels_dims, NPY_FLOAT32,
                                          current_batch->target_labels.data());

            npy_intp target_factors_dims[] = {
                (npy_intp)current_batch->target_factors.size(),
            };

            py::handle target_factors_numpy =
                PyArray_SimpleNewFromData(1, target_factors_dims, NPY_FLOAT32,
                                          current_batch->target_factors.data());

            npy_intp target_other_indices_dims[] = {
                (npy_intp)current_batch->target_other_indices.size(),
                2,
            };

            py::handle target_other_indices_numpy = PyArray_SimpleNewFromData(
                2, target_other_indices_dims, NPY_INT64,
                current_batch->target_other_indices.data());

            result["task"] = py::make_tuple(
                target_indices_numpy, target_labels_numpy, target_factors_numpy,
                target_other_indices_numpy);

        } else {
            // npy_intp label_indices_numpy_dims[] = {(npy_intp)
            // current_batch->label_indices.size()}; py::handle
            // label_indices_numpy = PyArray_SimpleNewFromData(1,
            // label_indices_numpy_dims, NPY_INT64,
            // current_batch->label_indices.data());

            // npy_intp labels_numpy_dims[] = {(npy_intp)
            // current_batch->labels.size()}; py::handle labels_numpy =
            // PyArray_SimpleNewFromData(1, labels_numpy_dims, NPY_FLOAT32,
            // current_batch->labels.data());

            // result["labeler"] = py::make_tuple(label_indices_numpy,
            // labels_numpy);
        }

        return result;
    }

    PTDatasetBatch* get_next_empty_batch() {
        PTDatasetBatch* result;
        empty_queue.wait_dequeue(result);
        return result;
    }

    void add_batch_to_empty(PTDatasetBatch* batch) {
        bool success = empty_queue.enqueue(batch);
        if (!success) {
            std::cerr << "Failed to enqueue" << std::endl;
            exit(-1);
        }
    }

    PTDatasetBatch* get_next_result_batch() {
        PTDatasetBatch* result;
        result_queue.wait_dequeue(result);
        return result;
    }

    void add_batch_to_result(PTDatasetBatch* batch) {
        bool success = result_queue.enqueue(batch);
        if (!success) {
            std::cerr << "Failed to enqueue" << std::endl;
            exit(-1);
        }
    }

   private:
    std::mt19937_64 rng;
    const PatientTimelineDataset& parent;
    const bool is_val;
    const float day_dropout;
    const float code_dropout;

    size_t num_threads;

    size_t done_threads;
    size_t next_index;
    std::vector<std::thread> threads;
    std::vector<PTDatasetBatch> batches;

    PTDatasetBatch* current_batch;
    moodycamel::BlockingConcurrentQueue<PTDatasetBatch*> result_queue;
    moodycamel::BlockingConcurrentQueue<PTDatasetBatch*> empty_queue;

    std::vector<std::pair<size_t, size_t>> indices;
    std::vector<std::pair<uint32_t, uint32_t>> patient_lengths;

    std::map<uint32_t, std::vector<std::pair<uint32_t, bool>>> label_map;
};

void* init_numpy() {
    import_array();
    return nullptr;
}

void register_patient2vec_extension(pybind11::module& root) {
    init_numpy();

    py::module m = root.def_submodule("patient2vec");
    m.def("create_info", create_info);

    py::class_<PatientTimelineDataset>(m, "PatientTimelineDataset")
        .def(py::init<const char*, const char*, const char*>())
        .def(py::init<const char*, const char*, const char*, py::tuple,
                      py::tuple>())
        .def("get_iterator", &PatientTimelineDataset::get_iterator,
             py::keep_alive<0, 1>())
        .def("num_batches", &PatientTimelineDataset::num_batches);

    py::class_<PTDatasetIterator>(m, "PTDatasetIterator")
        .def("__iter__",
             [](PTDatasetIterator& it) -> PTDatasetIterator& { return it; })
        .def("__next__", &PTDatasetIterator::next);
}
