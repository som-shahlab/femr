#include "clmbr_extension.h"

#include <pybind11/pybind11.h>

#include <optional>
#include <random>

#include "blockingconcurrentqueue.h"
#include "civil_day_caster.h"
#include "reader.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "numpy/numpy/ndarrayobject.h"

namespace py = pybind11;

namespace {

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
    int count;
    double current_mean;
    double variance;
};

std::string create_info(const char* timeline_path, const char* ontology_path,
                        absl::CivilDay train_start_date,
                        absl::CivilDay train_end_date,
                        absl::CivilDay val_start_date,
                        absl::CivilDay val_end_date, int min_patient_count) {
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

    std::vector<uint32_t> code_counts;
    std::vector<uint32_t> patient_code_set;

    size_t num_patients = reader.get_patient_ids().size();

    size_t ten_percent = num_patients / 10;

    size_t num_processed = 0;

    for (auto patient_id : reader.get_patient_ids()) {
        patient_code_set.clear();
        std::optional<uint32_t> train_start_age;
        std::optional<uint32_t> train_end_age;
        std::optional<uint32_t> val_start_age;
        std::optional<uint32_t> val_end_age;

        uint32_t train_length = 0;
        uint32_t val_length = 0;

        uint32_t valid_val_count = 0;
        uint32_t valid_train_count = 0;

        uint32_t last_age = 0;

        num_processed++;

        if (num_processed % ten_percent == 0) {
            std::cout << "Processed " << (100.0 * num_processed / num_patients)
                      << std::endl;
        }

        iter.process_patient(
            patient_id, [&](absl::CivilDay birth_day, uint32_t age,
                            const std::vector<uint32_t>& observations,
                            const std::vector<ObservationWithValue>&
                                observations_with_values) {
                if (!train_start_age) {
                    train_start_age =
                        std::max(train_start_date - birth_day,
                                 (absl::time_internal::cctz::diff_t)0);
                }
                if (!train_end_age) {
                    train_end_age =
                        std::max(train_end_date - birth_day,
                                 (absl::time_internal::cctz::diff_t)0);
                }
                if (!val_start_age) {
                    val_start_age =
                        std::max(val_start_date - birth_day,
                                 (absl::time_internal::cctz::diff_t)0);
                }
                if (!val_end_age) {
                    val_end_age =
                        std::max(val_end_date - birth_day,
                                 (absl::time_internal::cctz::diff_t)0);
                }

                if (age < *train_end_age) {
                    train_length++;

                    if (age != 0) {
                        uint32_t delta = age - last_age;
                        last_age = age;
                        add_stats(age_stats, age);
                        add_stats(delta_stats, delta);
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
                    }
                }

                if (age < *val_end_age) {
                    val_length++;
                }

                if (age >= *val_start_age && age < *val_end_age) {
                    valid_val_count++;
                }

                if (age >= *train_start_age && age < *train_end_age) {
                    valid_train_count++;
                }
            });

        if (train_length >= 3 && valid_train_count > 0) {
            std::sort(std::begin(patient_code_set), std::end(patient_code_set));
            auto unique_end = std::unique(std::begin(patient_code_set),
                                          std::end(patient_code_set));

            for (auto iter = std::begin(patient_code_set); iter != unique_end;
                 iter++) {
                uint32_t code = *iter;
                if (code >= code_counts.size()) {
                    code_counts.resize(code * 2 + 1);
                }
                code_counts[code]++;
            }

            train_patient_ids_with_length.push_back(
                std::make_pair(patient_id, train_length));
        }

        if (val_length >= 3 && valid_val_count > 0) {
            val_patient_ids_with_length.push_back(
                std::make_pair(patient_id, val_length));
        }
    }

    uint32_t codes_with_no_siblings = 0;
    uint32_t codes_with_no_path_to_root = 0;

    uint32_t root_code = ontologies.get_root_code();

    std::map<std::string, uint32_t> final_code_counts;
    std::vector<std::pair<uint32_t, uint32_t>> valid_codes;
    for (uint32_t code = 0; code < code_counts.size(); code++) {
        uint32_t count = code_counts[code];
        if (count != 0) {
            final_code_counts[std::to_string(code)] = count;
        }
        if (count < (uint32_t)min_patient_count) {
            continue;
        }

        auto all_parents = ontologies.get_all_parents(code);

        if (std::find(std::begin(all_parents), std::end(all_parents),
                      root_code) == std::end(all_parents)) {
            codes_with_no_path_to_root++;
            continue;
        }

        auto parents = ontologies.get_parents(code);

        bool has_sibling = false;

        for (auto parent : parents) {
            for (auto child : ontologies.get_children(parent)) {
                if (child != code && child < code_counts.size() &&
                    code_counts[child] >= (uint32_t)min_patient_count) {
                    has_sibling = true;
                }
            }
        }

        if (!has_sibling) {
            codes_with_no_siblings++;
            continue;
        }

        valid_codes.push_back(std::make_pair(-count, code));
    }

    std::cout << "Removed " << codes_with_no_siblings
              << " due to lack of siblings" << std::endl;
    std::cout << "Removed " << codes_with_no_path_to_root
              << " due to lack of a path to the root" << std::endl;

    std::sort(std::begin(valid_codes), std::end(valid_codes));

    std::cout << "Got " << valid_codes.size() << " valid codes " << std::endl;

    std::map<std::string, uint32_t> valid_code_map;

    for (size_t i = 0; i < valid_codes.size(); i++) {
        uint32_t code = valid_codes[i].second;
        valid_code_map[std::to_string(code)] = i;
    }

    nlohmann::json result;

    result["code_counts"] = final_code_counts;
    result["valid_code_map"] = valid_code_map;
    result["train_patient_ids_with_length"] = train_patient_ids_with_length;
    result["val_patient_ids_with_length"] = val_patient_ids_with_length;

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

std::vector<int64_t> convert_to_int(py::object data) {
    py::object int_data =
        py::reinterpret_steal<py::object>((PyObject*)PyArray_FromAny(
            data.ptr(), PyArray_DescrFromType(NPY_INT64), 1, 1,
            NPY_ARRAY_CARRAY_RO | NPY_ARRAY_FORCECAST, nullptr));
    const int64_t* ptr =
        (const int64_t*)PyArray_DATA((PyArrayObject*)int_data.ptr());
    size_t size = PyArray_SHAPE((PyArrayObject*)int_data.ptr())[0];
    std::vector<int64_t> result(ptr, ptr + size);
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

        // Check for and remove duplicates
        auto i = std::unique(std::begin(iter.second), std::end(iter.second), [](const std::pair<uint32_t, bool>& a, const std::pair<uint32_t, bool>& b) {
           return a.first == b.first; 
        });

        if (i != std::end(iter.second)) {
            std::cout<<"Had duplicates in the label map, this is almost certainly a bug" << std::endl;
            std::cout<<"Got duplicate for patient id " << iter.first << std::endl;

            abort();
        }
    }


    return label_map;
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

class PTDatasetIterator;
class PatientTimelineDataset {
   public:
    friend PTDatasetIterator;

    PatientTimelineDataset(const char* timelines_path,
                           const char* ontology_path, const char* info_path,
                           py::tuple train_data, py::tuple val_data)
        : PatientTimelineDataset(timelines_path, ontology_path, info_path) {
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
                if (code >= valid_recorded_date_code.size()) {
                    valid_recorded_date_code.resize(code * 2 + 1);
                }
                valid_recorded_date_code[code] = true;
            }

            nlohmann::json info;
            std::ifstream info_file(info_path);
            info_file >> info;

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

                if (key >= valid_code_map.size()) {
                    valid_code_map.resize((key + 1) * 2);
                }

                valid_code_map[key] = value;
            }

            num_valid_codes = info["valid_code_map"].size();

            auto decode_date = [](const std::string& date_str) {
                absl::CivilDay result;
                bool ok = absl::ParseCivilTime(date_str, &result);

                if (!ok) {
                    std::cout << "Could not parse the following date string "
                              << date_str << std::endl;
                    abort();
                }

                return result;
            };

            train_dates = std::make_pair(decode_date(info["train_start_date"]),
                                         decode_date(info["train_end_date"]));
            val_dates = std::make_pair(decode_date(info["val_start_date"]),
                                       decode_date(info["val_end_date"]));
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
    }

    std::unique_ptr<PTDatasetIterator> get_iterator(
        bool is_val, int batch_size, uint64_t seed, int threshold,
        float day_dropout = 0, float code_dropout = 0) const {
        return std::make_unique<PTDatasetIterator>(*this, is_val, batch_size,
                                                   seed, threshold, day_dropout,
                                                   code_dropout);
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

            result++;
        }

        return result;
    }

    int num_valid_codes;

   private:
    std::map<uint32_t, std::vector<std::pair<uint32_t, bool>>> train_map;
    std::map<uint32_t, std::vector<std::pair<uint32_t, bool>>> val_map;

    std::optional<uint32_t> get_valid_code(uint32_t code) const {
        if (code == ontologies.get_root_code()) {
            return std::nullopt;
        } else if (code >= valid_code_map.size()) {
            return std::nullopt;
        } else {
            return valid_code_map[code];
        }
    }

    bool is_valid_recorded_date_code(uint32_t code) const {
        if (code < valid_recorded_date_code.size()) {
            return valid_recorded_date_code[code];
        } else {
            return false;
        }
    }

    ExtractReader timelines;
    OntologyReader ontologies;

    std::vector<std::optional<uint32_t>> valid_code_map;
    std::vector<bool> valid_recorded_date_code;

    std::vector<std::pair<uint32_t, uint32_t>> train_lengths;
    std::vector<std::pair<uint32_t, uint32_t>> val_lengths;

    std::array<float, 50> rates;

    std::pair<absl::CivilDay, absl::CivilDay> train_dates;
    std::pair<absl::CivilDay, absl::CivilDay> val_dates;

    double age_mean, age_std, log_age_mean, log_age_std;
    double delta_mean, delta_std, log_delta_mean, log_delta_std;

    absl::flat_hash_set<uint32_t> bad_to_predict;
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

struct PTDatasetBatch {
    std::unique_ptr<ExtractReaderIterator> iter;

    size_t index;

    std::vector<uint32_t> positive_codes_per_day_features;
    std::vector<uint32_t> positive_codes_per_day;
    std::vector<uint32_t> negative_codes_per_day;
    std::vector<bool> drop_flags;

    std::vector<uint32_t> buffer;

    std::vector<int64_t> patient_ids;
    std::vector<int64_t> codes;
    std::vector<int64_t> offsets;
    std::vector<int64_t> codes1;
    std::vector<int64_t> offsets1;

    std::vector<std::array<int64_t, 2>> lengths;
    std::vector<std::array<float, 5>> day_info;

    std::vector<std::array<float, 200>> pos_encoding;

    std::vector<std::array<int64_t, 2>> target_indices;
    std::vector<float> targets;
    std::vector<float> target_seen;

    std::vector<std::array<int64_t, 2>> target_indices1;
    std::vector<float> targets1;
    std::vector<float> target1_seen;

    std::vector<int64_t> day_index;

    absl::flat_hash_set<uint32_t> seen_codes;

    std::vector<int64_t> label_indices;
    std::vector<int64_t> label_values;
};

class PTDatasetIterator {
   public:
    PTDatasetIterator(const PatientTimelineDataset& p, bool is_val_,
                      int batch_size, uint64_t seed, int threshold_,
                      float day_dropout_ = 0, float code_dropout_ = 0)
        : rng(seed),
          parent(p),
          is_val(is_val_),
          threshold(threshold_),
          day_dropout(day_dropout_),
          code_dropout(code_dropout_) {
        if (is_val) {
            patient_lengths = parent.val_lengths;
            dates = parent.val_dates;
            labels = parent.val_map;
        } else {
            patient_lengths = parent.train_lengths;
            dates = parent.train_dates;
            labels = parent.train_map;
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

            indices.push_back(std::make_pair(current_index, end_index));

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

        batch.target_indices.clear();
        batch.targets.clear();
        batch.target_seen.clear();
        batch.target_indices1.clear();
        batch.targets1.clear();
        batch.target1_seen.clear();

        batch.label_indices.clear();
        batch.label_values.clear();

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

        for (int i = 0; i < (end_index - start_index); i++) {
            batch.seen_codes.clear();

            uint32_t patient_id = patient_lengths.at(start_index + i).first;

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

            uint32_t length_with_drop = length - days_to_drop_for_patient;

            batch.drop_flags.clear();

            for (size_t i = 0; i < length; i++) {
                batch.drop_flags.push_back(i < days_to_drop_for_patient);
            }

            std::shuffle(std::begin(batch.drop_flags),
                         std::end(batch.drop_flags), batch_rng);

            batch.lengths.push_back(
                {(int64_t)batch.offsets.size(), (int64_t)(length_with_drop)});

            int32_t current_offset = 0;
            uint32_t last_age = 0;
            int64_t day_index = 0;

            std::optional<uint32_t> end_age;
            std::optional<uint32_t> start_age;

            std::vector<std::pair<uint32_t, bool>> current_labels;
            auto label_pointer = labels.find(patient_id);
            if (label_pointer != std::end(labels)) {
                current_labels = label_pointer->second;
            }
            auto current_label_iter = std::begin(current_labels);

            bool found = batch.iter->process_patient(
                patient_id, [&](absl::CivilDay birth_day, uint32_t age,
                                const std::vector<uint32_t>& observations,
                                const std::vector<ObservationWithValue>&
                                    observations_with_values) {
                    if (day_index > length) {
                        return;
                    }

                    if (!end_age) {
                        end_age =
                            std::max(dates.second - birth_day,
                                     (absl::time_internal::cctz::diff_t)0);
                    }

                    if (!start_age) {
                        start_age =
                            std::max(dates.first - birth_day,
                                     (absl::time_internal::cctz::diff_t)0);
                    }

                    // if (age >= *end_age) {
                    //     return;
                    // }

                    bool is_drop;

                    if (day_index == length) {
                        is_drop = false;
                    } else {
                        is_drop = batch.drop_flags[day_index];
                    }

                    batch.positive_codes_per_day.clear();
                    batch.positive_codes_per_day_features.clear();
                    batch.negative_codes_per_day.clear();

                    for (uint32_t code : observations) {
                        if (!parent.is_valid_recorded_date_code(code)) {
                            continue;
                        }

                        for (uint32_t subword :
                             parent.ontologies.get_subwords(code)) {
                            batch.positive_codes_per_day.push_back(subword);
                        }

                        if (code_dropout != 0 &&
                            !code_dropout_distribution(batch_rng)) {
                            for (uint32_t subword :
                                 parent.ontologies.get_subwords(code)) {
                                batch.positive_codes_per_day_features.push_back(
                                    subword);
                            }
                        }
                    }

                    for (auto code_with_value : observations_with_values) {
                        uint32_t code = code_with_value.code;

                        if (!parent.is_valid_recorded_date_code(code)) {
                            continue;
                        }

                        for (uint32_t subword :
                             parent.ontologies.get_subwords(code)) {
                            batch.positive_codes_per_day.push_back(subword);
                        }

                        if (code_dropout != 0 &&
                            !code_dropout_distribution(batch_rng)) {
                            for (uint32_t subword :
                                 parent.ontologies.get_subwords(code)) {
                                batch.positive_codes_per_day_features.push_back(
                                    subword);
                            }
                        }
                    }

                    std::sort(std::begin(batch.positive_codes_per_day),
                              std::end(batch.positive_codes_per_day));
                    auto last =
                        std::unique(std::begin(batch.positive_codes_per_day),
                                    std::end(batch.positive_codes_per_day));
                    batch.positive_codes_per_day.erase(
                        last, std::end(batch.positive_codes_per_day));

                    if (code_dropout != 0) {
                        std::sort(
                            std::begin(batch.positive_codes_per_day_features),
                            std::end(batch.positive_codes_per_day_features));
                        auto last = std::unique(
                            std::begin(batch.positive_codes_per_day_features),
                            std::end(batch.positive_codes_per_day_features));
                        batch.positive_codes_per_day_features.erase(
                            last,
                            std::end(batch.positive_codes_per_day_features));
                    }

                    get_negative_codes(parent.ontologies,
                                       batch.positive_codes_per_day,
                                       batch.negative_codes_per_day);

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
                            double delta_double = (age - last_age);
                            double log_delta = std::log(1 + delta_double);
                            batch.day_info.push_back({
                                (float)((age_double - parent.age_mean) /
                                        parent.age_std),
                                (float)((log_age - parent.log_age_mean) /
                                        parent.log_age_std),
                                (float)((delta_double - parent.delta_mean) /
                                        parent.delta_std),
                                (float)((log_delta - parent.log_delta_mean) /
                                        parent.log_delta_std),
                                0,
                            });
                        }

                        std::array<float, 200> position;

                        for (int pos_i = 0; pos_i < 50; pos_i++) {
                            position[pos_i * 4 + 0] =
                                std::sin(current_offset * parent.rates[pos_i]);
                            position[pos_i * 4 + 1] =
                                std::cos(current_offset * parent.rates[pos_i]);
                            position[pos_i * 4 + 2] =
                                std::sin(age / 365 * parent.rates[pos_i]);
                            position[pos_i * 4 + 3] =
                                std::cos(age / 365 * parent.rates[pos_i]);
                        }

                        batch.pos_encoding.push_back(position);

                        batch.offsets.push_back(batch.codes.size());
                        batch.offsets1.push_back(batch.codes1.size());

                        bool added_one = false;
                        bool added_one1 = false;

                        if (code_dropout != 0) {
                            for (uint32_t code :
                                 batch.positive_codes_per_day_features) {
                                if (auto valid_code =
                                        parent.get_valid_code(code)) {
                                    if (*valid_code >= threshold) {
                                        batch.codes1.push_back(*valid_code -
                                                               threshold);
                                        added_one1 = true;
                                    } else {
                                        batch.codes.push_back(*valid_code);
                                        added_one = true;
                                    }
                                }
                            }
                        } else {
                            for (uint32_t code : batch.positive_codes_per_day) {
                                if (auto valid_code =
                                        parent.get_valid_code(code)) {
                                    if (*valid_code >= threshold) {
                                        batch.codes1.push_back(*valid_code -
                                                               threshold);
                                        added_one1 = true;
                                    } else {
                                        batch.codes.push_back(*valid_code);
                                        added_one = true;
                                    }
                                }
                            }
                        }

                        if (!added_one) {
                            batch.codes.push_back(threshold);
                        }

                        if (!added_one1) {
                            batch.codes1.push_back(parent.num_valid_codes -
                                                   threshold);
                        }

                        last_age = age;
                    }

                    if (!is_drop) {
                        current_offset += 1;
                    }

                    if (!is_drop &&
                        current_label_iter != std::end(current_labels) &&
                        current_label_iter->first == day_index + 1) {
                        batch.label_indices.push_back((current_offset - 1) +
                                                      i * max_length);
                        batch.label_values.push_back(
                            current_label_iter->second);
                        current_label_iter++;
                    }

                    if (current_offset > 2 && age >= *start_age && !is_drop) {
                        for (uint32_t code : batch.positive_codes_per_day) {
                            if (parent.bad_to_predict.find(code) !=
                                std::end(parent.bad_to_predict)) {
                                continue;
                            }
                            if (auto valid_code = parent.get_valid_code(code)) {
                                float seen =
                                    batch.seen_codes.find(*valid_code) !=
                                    batch.seen_codes.end();
                                if (*valid_code >= threshold) {
                                    batch.target_indices1.push_back(
                                        {(current_offset - 2) + i * max_length,
                                         *valid_code - threshold});
                                    batch.targets1.push_back(1);
                                    batch.target1_seen.push_back(seen);
                                } else {
                                    batch.target_indices.push_back(
                                        {(current_offset - 2) + i * max_length,
                                         *valid_code});
                                    batch.targets.push_back(1);
                                    batch.target_seen.push_back(seen);
                                }
                            }
                        }

                        for (uint32_t code : batch.negative_codes_per_day) {
                            if (parent.bad_to_predict.find(code) !=
                                std::end(parent.bad_to_predict)) {
                                continue;
                            }
                            if (auto valid_code = parent.get_valid_code(code)) {
                                float seen =
                                    batch.seen_codes.find(*valid_code) !=
                                    batch.seen_codes.end();
                                if (*valid_code >= threshold) {
                                    batch.target_indices1.push_back(
                                        {(current_offset - 2) + i * max_length,
                                         *valid_code - threshold});
                                    batch.targets1.push_back(0);
                                    batch.target1_seen.push_back(seen);
                                } else {
                                    batch.target_indices.push_back(
                                        {(current_offset - 2) + i * max_length,
                                         *valid_code});
                                    batch.targets.push_back(0);
                                    batch.target_seen.push_back(seen);
                                }
                            }
                        }
                    }

                    if (!is_drop) {
                        if (code_dropout != 0) {
                            for (uint32_t code :
                                 batch.positive_codes_per_day_features) {
                                if (auto valid_code =
                                        parent.get_valid_code(code)) {
                                    batch.seen_codes.insert(*valid_code);
                                }
                            }
                        } else {
                            for (uint32_t code : batch.positive_codes_per_day) {
                                if (auto valid_code =
                                        parent.get_valid_code(code)) {
                                    batch.seen_codes.insert(*valid_code);
                                }
                            }
                        }
                    }

                    day_index++;
                });

            if (day_index != length + 1 && day_index != length) {
                std::cout
                    << "Day index should count up to length or length + 1?"
                    << patient_id << " " << day_index << " " << length
                    << std::endl;
                abort();
            }

            if (!found) {
                std::cout << "Could not find patient_id " << patient_id
                          << std::endl;
                abort();
            }
        }
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

        npy_intp target_indices_numpy_dims[] = {
            (npy_intp)current_batch->target_indices.size(), 2};
        py::handle target_indices_numpy =
            PyArray_SimpleNewFromData(2, target_indices_numpy_dims, NPY_INT64,
                                      current_batch->target_indices.data());

        npy_intp targets_numpy_dims[] = {
            (npy_intp)current_batch->targets.size()};
        py::handle targets_numpy = PyArray_SimpleNewFromData(
            1, targets_numpy_dims, NPY_FLOAT32, current_batch->targets.data());

        npy_intp target_seen_numpy_dims[] = {
            (npy_intp)current_batch->target_seen.size()};
        py::handle target_seen_numpy =
            PyArray_SimpleNewFromData(1, target_seen_numpy_dims, NPY_FLOAT32,
                                      current_batch->target_seen.data());

        npy_intp target_indices1_numpy_dims[] = {
            (npy_intp)current_batch->target_indices1.size(), 2};
        py::handle target_indices1_numpy =
            PyArray_SimpleNewFromData(2, target_indices1_numpy_dims, NPY_INT64,
                                      current_batch->target_indices1.data());

        npy_intp targets1_numpy_dims[] = {
            (npy_intp)current_batch->targets1.size()};
        py::handle targets1_numpy =
            PyArray_SimpleNewFromData(1, targets1_numpy_dims, NPY_FLOAT32,
                                      current_batch->targets1.data());

        npy_intp target1_seen_numpy_dims[] = {
            (npy_intp)current_batch->target1_seen.size()};
        py::handle target1_seen_numpy =
            PyArray_SimpleNewFromData(1, target1_seen_numpy_dims, NPY_FLOAT32,
                                      current_batch->target1_seen.data());

        result["task"] = py::make_tuple(
            target_indices_numpy, targets_numpy, target_seen_numpy,
            target_indices1_numpy, targets1_numpy, target1_seen_numpy);

        npy_intp label_indices_numpy_dims[] = {
            (npy_intp)current_batch->label_indices.size()};
        py::handle label_indices_numpy =
            PyArray_SimpleNewFromData(1, label_indices_numpy_dims, NPY_INT64,
                                      current_batch->label_indices.data());

        npy_intp label_values_numpy_dims[] = {
            (npy_intp)current_batch->label_values.size()};
        py::handle label_values_numpy =
            PyArray_SimpleNewFromData(1, label_values_numpy_dims, NPY_INT64,
                                      current_batch->label_values.data());

        result["label"] =
            py::make_tuple(label_indices_numpy, label_values_numpy);

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
    const uint32_t threshold;
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

    std::map<uint32_t, std::vector<std::pair<uint32_t, bool>>> labels;

    std::pair<absl::CivilDay, absl::CivilDay> dates;
};

void* init_numpy() {
    import_array();
    return nullptr;
}
}  // namespace

void register_clmbr_extension(pybind11::module& root) {
    init_numpy();

    py::module m = root.def_submodule("clmbr");
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
