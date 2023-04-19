# include "dataloader_extension.hh"

#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace py = pybind11;

#include <algorithm>
#include <boost/optional/optional_io.hpp>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <queue>
#include <random>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "clmbr_dictionary.hh"
#include "constdb.hh"
#include "create_dictionary.hh"
#include "create_survival_dictionary.hh"
#include "database.hh"
#include "flatmap.hh"
#include "npy.hh"
#include "pybind11/eigen/tensor.h"
#include "survival.hh"

const bool SPARSE_FALLBACK = false;

class Task {
   public:
    virtual ~Task(){};

    virtual const std::vector<uint32_t>& get_patient_offsets() = 0;
    virtual void start_batch() = 0;
    virtual void start_patient(const Patient& p) = 0;
    virtual bool needs_exact() const { return false; }

    virtual bool add_event_data(int current_year, uint32_t current_age,
                                const std::vector<uint32_t>& next_features,
                                boost::optional<uint32_t> next_age,
                                bool actually_add,
                                bool using_dropout = false) = 0;

    virtual void prepare_batch_data(uint32_t num_representations) = 0;
    virtual py::dict get_batch_data() const = 0;
};

uint32_t round_to_nearest_bin(uint32_t value, uint32_t skip = 1) {
    uint32_t result = 1;
    while (value > result) {
        result <<= skip;
    }
    assert(result >= value);
    return result;
}

class LabeledPatientsTask : public Task {
   public:
    LabeledPatientsTask(json config, PatientDatabase& data) {
        labeler_type = config["labeler_type"];
        for (json label : config["labels"]) {
            uint64_t patient_id = label[0];
            uint32_t patient_offset = *data.get_patient_offset(patient_id);
            uint32_t age_in_minutes = label[1];
            json value = label[2];
            labels[patient_offset].push_back(std::make_pair(age_in_minutes, value));
        }

        for (auto& entry : labels) {
            patient_offsets.push_back(entry.first);
            std::sort(std::begin(entry.second), std::end(entry.second));
        }
    }

    const std::vector<uint32_t>& get_patient_offsets() override {
        return patient_offsets;
    }
    void start_batch() override { batch_labels.clear(); }
    bool needs_exact() const override { return true; }

    void start_patient(const Patient& p) override {
        current_patient_iter = labels.find(p.patient_offset);
        if (current_patient_iter == std::end(labels)) {
            throw std::runtime_error("Trying to process an invalid patient");
        }

        current_label_iter = std::begin(current_patient_iter->second);
    }

    bool add_event_data(int current_year, uint32_t current_age,
                        const std::vector<uint32_t>& next_features,
                        boost::optional<uint32_t> next_age, bool actually_add,
                        bool using_dropout) override {
        boost::optional<std::pair<uint32_t, json>> label_to_add;

        while (current_label_iter != std::end(current_patient_iter->second) &&
               (!next_age || current_label_iter->first < *next_age)) {
            label_to_add = *current_label_iter;

            current_label_iter++;
        }

        if (actually_add && label_to_add) {
            if (label_to_add->first < current_age) {
                throw std::runtime_error(
                    absl::StrCat("This should not be possible ", current_age,
                                 " ", label_to_add->first, " ", *next_age));
            }

            if (labeler_type == "survival") {
                uint32_t invalid_delta = label_to_add->first - current_age;
                if (!using_dropout && invalid_delta > 60 * 24) {
                    std::cout << "This should never happen " << invalid_delta
                              << " " << current_patient_iter->first << " "
                              << label_to_add->first << " " << current_age
                              << " " << next_age << std::endl;
                }
                if (using_dropout) {
                    label_to_add->first = current_age;
                }
            }
            batch_labels.push_back(*label_to_add);
        }

        return (bool)label_to_add;
    }

    Eigen::Tensor<uint32_t, 1> final_batch_ages;

    Eigen::Tensor<float, 1> final_batch_labels;

    Eigen::Tensor<bool, 1> final_batch_censor;
    Eigen::Tensor<float, 1> final_batch_event_times;
    virtual void prepare_batch_data(uint32_t num_representations) override {
        final_batch_ages = Eigen::Tensor<uint32_t, 1>(num_representations);
        final_batch_ages.setConstant(0);

        for (uint32_t i = 0; i < batch_labels.size(); i++) {
            final_batch_ages(i) = batch_labels[i].first;
        }

        if (labeler_type == "boolean") {
            final_batch_labels = Eigen::Tensor<float, 1>(num_representations);
            final_batch_labels.setConstant(0);

            for (uint32_t i = 0; i < batch_labels.size(); i++) {
                final_batch_labels(i) = batch_labels[i].second;
            }
        } else if (labeler_type == "survival") {
            final_batch_censor = Eigen::Tensor<bool, 1>(num_representations);
            final_batch_censor.setConstant(true);

            final_batch_event_times =
                Eigen::Tensor<float, 1>(num_representations);
            final_batch_event_times.setConstant(0);

            for (uint32_t i = 0; i < batch_labels.size(); i++) {
                final_batch_censor(i) = batch_labels[i].second["is_censored"];
                final_batch_event_times(i) =
                    batch_labels[i].second["event_time"].get<uint32_t>() -
                    batch_labels[i].first;
            }
        }
    }

    py::dict get_batch_data() const override {
        py::dict result;
        result["label_ages"] = final_batch_ages;
        if (labeler_type == "boolean") {
            result["labels"] = final_batch_labels;
        } else if (labeler_type == "survival") {
            result["is_censor"] = final_batch_censor;
            result["event_times"] = final_batch_event_times;
        }
        return result;
    }

   private:
    std::string labeler_type;

    absl::flat_hash_map<uint32_t, std::vector<std::pair<uint32_t, json>>>
        labels;
    std::vector<uint32_t> patient_offsets;

    std::vector<std::pair<uint32_t, json>> batch_labels;

    absl::flat_hash_map<uint32_t,
                        std::vector<std::pair<uint32_t, json>>>::const_iterator
        current_patient_iter;
    std::vector<std::pair<uint32_t, json>>::const_iterator current_label_iter;
};

class CLMBRTask : public Task {
   public:
    CLMBRTask(json config, PatientDatabase& data) {
        // Might be empty, in which case we train on everyone
        std::vector<uint64_t> patient_ids = config.value("patient_ids", std::vector<uint64_t>());
        for (uint64_t patient_id : patient_ids) {
            patient_offsets.push_back(*data.get_patient_offset(patient_id));
        }
        vocab_size = config["vocab_size"];
    }

    const std::vector<uint32_t>& get_patient_offsets() override {
        return patient_offsets;
    }

    void start_batch() override { batch_labels.clear(); }

    void start_patient(const Patient& p) override {}

    bool add_event_data(int current_year, uint32_t current_age,
                        const std::vector<uint32_t>& next_features,
                        boost::optional<uint32_t> next_age, bool actually_add,
                        bool using_dropout) override {
        if (next_features.size() == 0) {
            return false;
        }

        if (next_features.size() != 1) {
            throw std::runtime_error("Only supports one for right now");
        }

        uint32_t next_feature = next_features[0];

        if (next_feature >= vocab_size) {
            return false;
        }

        if (!next_age) {
            return false;
        }

        if (*next_age < 2 * 60 * 24) {
            return false;
        }

        if (actually_add) {
            batch_labels.push_back(next_feature);
        }
        return true;
    }

    Eigen::Tensor<uint32_t, 1> final_batch_labels;
    virtual void prepare_batch_data(uint32_t num_representations) override {
        final_batch_labels = Eigen::Tensor<uint32_t, 1>(num_representations);
        final_batch_labels.setConstant(0);

        for (uint32_t i = 0; i < batch_labels.size(); i++) {
            final_batch_labels(i) = batch_labels[i];
        }
    }

    py::dict get_batch_data() const override {
        py::dict result;
        result["labels"] = final_batch_labels;

        return result;
    }

   private:
    uint32_t vocab_size;

    std::vector<uint32_t> patient_offsets;

    std::vector<uint32_t> batch_labels;
};

class SurvivalCLMBRTask : public Task {
   public:
    SurvivalCLMBRTask(json config, PatientDatabase& data, Ontology& ontology) {
        // Might be empty, in which case we train on everyone
        std::vector<uint64_t> patient_ids = config.value("patient_ids", std::vector<uint64_t>());
        for (uint64_t patient_id : patient_ids) {
            patient_offsets.push_back(*data.get_patient_offset(patient_id));
        }
        json survival_dict = config["survival_dict"];
        survival_dictionary = get_mapping(ontology, survival_dict["codes"]);
        vocab_size = survival_dict["codes"].size();
        time_bins = survival_dict["time_bins"].get<std::vector<uint32_t>>();
    }

    const std::vector<uint32_t>& get_patient_offsets() override {
        return patient_offsets;
    }

    void start_batch() override {
        events_per_rep.clear();
        total_events = 0;
    }

    void start_patient(const Patient& p) override {
        calculator.preprocess_patient(p, survival_dictionary);
        last_prediction_age = 0;
    }

    bool add_event_data(int current_year, uint32_t current_age,
                        const std::vector<uint32_t>& next_features,
                        boost::optional<uint32_t> next_age, bool actually_add,
                        bool using_dropout) override {
        if (!should_make_prediction(last_prediction_age, current_age, next_age,
                                    current_year)) {
            return false;
        }

        auto entry = calculator.get_times_for_event(current_age);
        if (entry.second.size() == 0) {
            return false;
        }

        last_prediction_age = current_age;

        if (actually_add) {
            total_events += entry.second.size();
            events_per_rep.push_back(entry);
        }

        return true;
    }

    Eigen::Tensor<uint32_t, 2> batch_event_indices;
    Eigen::Tensor<uint32_t, 1> batch_event_offsets;
    Eigen::Tensor<float, 1> batch_censor_log_times;

    Eigen::Tensor<uint32_t, 1> batch_event_codes;
    Eigen::Tensor<float, 1> batch_event_log_times;

    Eigen::Tensor<float, 3> dense_log_times;
    Eigen::Tensor<bool, 3> dense_is_event;

    void prepare_batch_data(uint32_t num_representations) override {
        uint32_t num_batch_events = round_to_nearest_bin(total_events, 2);

        batch_event_indices = Eigen::Tensor<uint32_t, 2>(num_batch_events, 2);
        for (uint32_t i = 0; i < num_batch_events; i++) {
            batch_event_indices(i, 0) = num_representations * time_bins.size();
            batch_event_indices(i, 1) = vocab_size;
        }

        dense_log_times = Eigen::Tensor<float, 3>(num_representations,
                                                  time_bins.size(), vocab_size);
        dense_log_times.setConstant(-std::numeric_limits<float>::infinity());

        dense_is_event = Eigen::Tensor<bool, 3>(num_representations,
                                                time_bins.size(), vocab_size);
        dense_is_event.setConstant(false);

        batch_event_offsets = Eigen::Tensor<uint32_t, 1>(
            num_representations * time_bins.size() + 1);
        batch_event_offsets.setConstant(num_batch_events * time_bins.size());

        batch_censor_log_times =
            Eigen::Tensor<float, 1>(num_representations * time_bins.size());
        batch_censor_log_times.setConstant(
            -std::numeric_limits<float>::infinity());

        batch_event_codes =
            Eigen::Tensor<uint32_t, 1>(num_batch_events * time_bins.size());
        batch_event_codes.setConstant(vocab_size);

        batch_event_log_times =
            Eigen::Tensor<float, 1>(num_batch_events * time_bins.size());
        batch_event_log_times.setConstant(
            std::numeric_limits<float>::quiet_NaN());

        uint32_t event_index = 0;
        uint32_t offset_index = 0;
        for (uint32_t rep = 0; rep < events_per_rep.size(); rep++) {
            auto& entry = events_per_rep[rep];
            uint32_t censor = entry.first;
            auto& events = entry.second;

            std::sort(std::begin(events), std::end(events),
                      [&](const auto& a, const auto& b) {
                          return a.second < b.second;
                      });

            uint32_t found_events_for_bin = 0;

            for (uint32_t time_bin = 0; time_bin < time_bins.size();
                 time_bin++) {
                uint32_t start = time_bins[time_bin];

                if (censor < start) {
                    batch_censor_log_times(rep * time_bins.size() + time_bin) =
                        -std::numeric_limits<float>::infinity();

                    batch_event_offsets(rep * time_bins.size() + time_bin) =
                        offset_index;
                } else {
                    uint32_t time_in_bin = censor - start;

                    uint32_t end;
                    if (time_bin == time_bins.size() - 1) {
                        end = std::numeric_limits<uint32_t>::max();
                    } else {
                        end = time_bins[time_bin + 1];
                    }

                    time_in_bin = std::min(time_in_bin, end - start);

                    float log_time_in_bin = std::log2(time_in_bin);

                    for (uint32_t i = 0; i < vocab_size; i++) {
                        dense_log_times(rep, time_bin, i) = log_time_in_bin;
                    }

                    batch_censor_log_times(rep * time_bins.size() + time_bin) =
                        log_time_in_bin;

                    batch_event_offsets(rep * time_bins.size() + time_bin) =
                        offset_index;

                    for (const auto& event : events) {
                        if (event.first >= end) {
                            continue;
                        }

                        float log_time =
                            -std::numeric_limits<float>::infinity();
                        if (event.first >= start) {
                            batch_event_indices(event_index, 0) =
                                rep * time_bins.size() + time_bin;
                            batch_event_indices(event_index, 1) = event.second;

                            found_events_for_bin++;
                            event_index++;

                            log_time = std::log2(event.first - start);
                            dense_is_event(rep, time_bin, event.second) = true;
                        }

                        dense_log_times(rep, time_bin, event.second) = log_time;

                        batch_event_codes(offset_index) = event.second;
                        batch_event_log_times(offset_index) = log_time;
                        offset_index++;
                    }
                }
            }

            if (found_events_for_bin != events.size()) {
                throw std::runtime_error("Failed?");
            }
        }

        if (event_index != total_events) {
            throw std::runtime_error(absl::StrCat(
                "Main one failed? ", event_index, " ", total_events));
        }
        for (uint32_t index = events_per_rep.size() * time_bins.size();
             index < num_representations * time_bins.size() + 1; index++) {
            batch_event_offsets(index) = offset_index;
        }
    }

    py::dict get_batch_data() const override {
        py::list sparse_time;
        sparse_time.append(batch_event_offsets);
        sparse_time.append(batch_censor_log_times);
        sparse_time.append(batch_event_codes);
        sparse_time.append(batch_event_log_times);

        py::dict result;
        result["num_valid"] = total_events;
        result["event_indices"] = batch_event_indices;
        result["sparse_time"] = sparse_time;
        result["dense_is_event"] = dense_is_event;
        result["dense_log_times"] = dense_log_times;

        return result;
    }

   private:
    std::vector<std::pair<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>>>
        events_per_rep;
    size_t total_events;

    uint32_t last_prediction_age;
    SurvivalCalculator calculator;

    std::vector<uint32_t> patient_offsets;

    FlatMap<std::vector<uint32_t>> survival_dictionary;
    uint32_t vocab_size;
    std::vector<uint32_t> time_bins;
};

std::unique_ptr<Task> create_task(json config, PatientDatabase& data, Ontology& ontology) {
    std::string type = config["type"];
    if (type == "labeled_patients") {
        return std::make_unique<LabeledPatientsTask>(config, data);
    } else if (type == "clmbr") {
        return std::make_unique<CLMBRTask>(config, data);
    } else if (type == "survival_clmbr") {
        return std::make_unique<SurvivalCLMBRTask>(config, data, ontology);
    }

    throw std::runtime_error("Invalid task type " + type);
}

class FeatureLookup {
   public:
    FeatureLookup(const json& data, uint32_t vocab_size, bool is_hierarchical,
                  Ontology& ontology, PatientDatabase& database) {
        const json::array_t* actual_data;
        this->is_hierarchical = is_hierarchical;
        this->vocab_size = vocab_size;

        if (is_hierarchical) {
            actual_data =
                data["ontology_rollup"].get_ptr<const json::array_t*>();
        } else {
            actual_data = data["regular"].get_ptr<const json::array_t*>();
        };

        absl::flat_hash_map<uint32_t, uint32_t> code_features_temp;

        uint32_t missing = 0;
        uint32_t searched = 0;

        for (uint32_t i = 0; i < vocab_size; i++) {
            DictEntry entry = (*actual_data)[i].get<DictEntry>();
            if (entry.type == DictEntryType::UNUSED) {
                continue;
            }

            searched++;
            auto possible_code = ontology.get_dictionary().find(entry.code_string);
            if (!possible_code.has_value()) {
                missing++;
                continue;
            }

            uint32_t code = *possible_code;

            switch (entry.type) {
                case DictEntryType::CODE:
                    code_features_temp[code] = i;
                    break;

                case DictEntryType::TEXT: {
                    auto possible_text = database.get_shared_text_dictionary().find(entry.text_string);
                    if (!possible_text) {
                        missing++;
                    } else {
                        text_features[std::make_pair(code,
                                                 *possible_text)] = i;
                    }
                    break;
                }

                case DictEntryType::NUMERIC:
                    numeric_features[code].push_back(
                        std::make_tuple(entry.val_start, entry.val_end, i));
                    break;

                case DictEntryType::UNUSED:
                    break;
            }
        }

        std::cout<<"When mapping codes, dropped " << missing << " out of " << searched << std::endl;

        for (uint32_t code = 0; code < ontology.get_dictionary().size();
             code++) {
            if (is_hierarchical) {
                std::vector<uint32_t> features;
                for (uint32_t parent : ontology.get_all_parents(code)) {
                    auto iter = code_features_temp.find(parent);
                    if (iter != std::end(code_features_temp)) {
                        features.push_back(iter->second);
                    }
                }

                if (features.size() > 0) {
                    std::sort(std::begin(features), std::end(features));
                    code_features[code] = features;
                }
            } else {
                auto iter = code_features_temp.find(code);
                if (iter != std::end(code_features_temp)) {
                    code_features[code] = {iter->second};
                }
            }
        }
    }

    std::vector<uint32_t> get_feature_codes(const Event& event) const {
        switch (event.value_type) {
            case ValueType::NONE: {
                auto iter = code_features.find(event.code);
                if (iter == std::end(code_features)) {
                    return {};
                }
                return iter->second;
            }

            case ValueType::NUMERIC: {
                auto iter = numeric_features.find(event.code);
                if (iter == std::end(numeric_features)) {
                    return {};
                }
                for (const auto& item : iter->second) {
                    if (event.numeric_value >= std::get<0>(item) &&
                        event.numeric_value < std::get<1>(item)) {
                        return {std::get<2>(item)};
                    }
                }
                return {};
            }

            case ValueType::SHARED_TEXT: {
                auto iter = text_features.find(
                    std::make_pair(event.code, event.text_value));
                if (iter == std::end(text_features)) {
                    return {};
                }
                return {iter->second};
            }

            case ValueType::UNIQUE_TEXT:
                return {};

            default:
                throw std::runtime_error("Invalid value type?");
        }
    }

    bool is_hierarchical;
    uint32_t vocab_size;

   private:
    absl::flat_hash_map<std::pair<uint32_t, uint32_t>, uint32_t> text_features;
    absl::flat_hash_map<uint32_t,
                        std::vector<std::tuple<double, double, uint32_t>>>
        numeric_features;
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>> code_features;
};

json read_file(std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    return json::from_msgpack(file);
}

class SplitMix64 {
   public:
    SplitMix64(uint64_t seed) { x = seed; }
    uint64_t next() {
        uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
        z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
        return z ^ (z >> 31);
    }

    double next_float() {
        return ((double)next()) / (double)std::numeric_limits<uint64_t>::max();
    }

   private:
    uint64_t x;
};

class BatchCreator {
   public:
    BatchCreator(PatientDatabase& _data, const json& config,
                 double _token_dropout = 0)
        : data(_data),
          iter(data.iterator()),
          lookup(config["transformer"]["dictionary"],
                 config["transformer"]["vocab_size"],
                 config["transformer"].value("is_hierarchical", false),
                 data.get_ontology(), data),
          task(create_task(config["task"], data, data.get_ontology())),
          rng(config["seed"]),
          token_dropout(_token_dropout) {
        uint32_t min_size = config["transformer"]["min_size"];
        uint32_t max_size = config["transformer"]["max_size"];

        age_mean = config["transformer"]["dictionary"]["age_stats"]["mean"];
        age_std = config["transformer"]["dictionary"]["age_stats"]["std"];

        (void) data.get_patient_id(0);
        patient_ids = Eigen::Tensor<uint64_t, 1>(1 << (max_size - min_size));
        offsets = Eigen::Tensor<uint32_t, 1>(1 << (max_size - min_size));
        if (lookup.is_hierarchical) {
            if (SPARSE_FALLBACK) {
                bad_tokens =
                    Eigen::Tensor<float, 2>(1 << max_size, lookup.vocab_size);
            }
        } else {
            tokens = Eigen::Tensor<uint32_t, 1>(1 << max_size);
        }
        valid_tokens = Eigen::Tensor<bool, 1>(1 << max_size);
        ages = Eigen::Tensor<float, 1>(1 << max_size);
        normalized_ages = Eigen::Tensor<float, 1>(1 << max_size);
        is_note_embedding = Eigen::Tensor<bool, 1>(1 << max_size);

        std::string note_path =
            config["transformer"].value("note_embedding_data", "");

        if (!note_path.empty()) {
            reader.emplace(note_path.c_str(), true);
        }
    }

    void start_batch(uint32_t max_length) {
        batch_index = 0;
        label_indices.clear();
        task->start_batch();
        this->max_length = max_length;

        patient_ids.setConstant(0);
        offsets.setConstant(0);

        valid_tokens.setConstant(false);
        if (lookup.is_hierarchical) {
            if (SPARSE_FALLBACK) {
                bad_tokens.setConstant(0);
            }
            sparse_token_indices_list.clear();

        } else {
            tokens.setConstant(0);
        }
        ages.setConstant(0);
        normalized_ages.setConstant(0);

        is_note_embedding.setConstant(false);

        current_note_offset = 0;
        current_note_embedding_bytes.clear();
    }

    void add_patient(uint32_t patient_offset, uint32_t offset,
                     bool actually_add = true) {
        if (batch_index >= offsets.size()) {
            throw std::runtime_error(
                "This should not be possible to go over the batch size?");
        }
        const Patient& p = iter.get_patient(patient_offset);

        patient_ids(batch_index) = data.get_patient_id(patient_offset);

        if (!task->needs_exact() && offset != 0) {
            if (token_dropout != 0) {
                uint32_t random_offset = rng.next_float() * offset;
                offset = random_offset;
            } else {
                offset = 0;
            }
        }
        offsets(batch_index) = offset;

        codes_seen_today.clear();
        uint32_t last_age = 0;

        uint32_t total_length = 0;

        task->start_patient(p);

        std::vector<int64_t> current_indices;

        if (reader) {
            auto entry = reader->get_int(p.patient_offset * 2);

            if (entry.second != 0) {
                std::string event_indices(entry.first, entry.second);
                std::istringstream event_indices_stream(event_indices);

                std::vector<unsigned long> shape;

                bool fortran_order;

                npy::LoadArrayFromNumpy(event_indices_stream, shape,
                                        fortran_order, current_indices);

                if (false) {
                    std::cout << "Reading event indices ... " << std::endl;
                    std::cout << "Shape: ";
                    for (const auto& a : shape) {
                        std::cout << a << " ";
                    }
                    std::cout << std::endl;

                    std::cout << "Values: ";
                    for (const auto& a : current_indices) {
                        std::cout << a << " ";
                    }
                    std::cout << std::endl;
                }

                auto embedding_entry = reader->get_int(p.patient_offset * 2 + 1);

                std::string embedding_bytes(embedding_entry.first,
                                            embedding_entry.second);

                std::istringstream embedding_bytes_stream(embedding_bytes);

                std::vector<char> bytes;

                npy::LoadArrayFromNumpy(embedding_bytes_stream, shape,
                                        fortran_order, bytes);

                if (bytes.size() != (768 * current_indices.size() * 2)) {
                    throw std::runtime_error(
                        absl::StrCat("Does not match up? ", bytes.size(), " ",
                                     current_indices.size()));
                }

                current_note_embedding_bytes.insert(
                    std::end(current_note_embedding_bytes), std::begin(bytes),
                    std::end(bytes));
            }
        }

        auto note_iter = std::begin(current_indices);

        for (size_t event_index = 0; event_index < p.events.size();
             event_index++) {
            const Event& event = p.events[event_index];

            if (token_dropout != 0 && rng.next_float() < token_dropout) {
                continue;
            }

            while (note_iter != std::end(current_indices) &&
                   (size_t)(*note_iter) < event_index) {
                note_iter++;
            }

            bool is_note = (note_iter != std::end(current_indices)) &&
                           ((size_t)*note_iter == event_index);

            if (false && is_note) {
                std::cout << "What in the world " << p.patient_offset << " "
                          << event_index << " " << int(event.value_type) << " "
                          << event.text_value << std::endl;
                std::cout << "Lol?" << std::endl;
                const Dictionary* dict;
                if (event.value_type == ValueType::UNIQUE_TEXT) {
                    dict = data.get_unique_text_dictionary();
                } else {
                    dict = &(data.get_shared_text_dictionary());
                }
                std::string_view item = (*dict)[event.text_value];
                std::cout << "Got it next " << item.size() << std::endl;
                std::cout << "Sure " << std::string(item) << std::endl;
            }

            std::vector<uint32_t> features;
            if (!is_note) {
                features = lookup.get_feature_codes(event);
            } else {
                features.push_back((note_iter - std::begin(current_indices)) +
                                   current_note_offset);
            }
            if (features.size() == 0) {
                continue;
            }

            if (event.start_age_in_minutes / (60 * 24) !=
                last_age / (60 * 24)) {
                codes_seen_today.clear();
            }

            if (!is_note) {
                bool all_seen_before = true;
                for (uint32_t feature : features) {
                    if (!codes_seen_today.count(feature)) {
                        all_seen_before = false;
                        codes_seen_today.insert(feature);
                    }
                }

                if (all_seen_before) {
                    continue;
                }
            }

            int32_t index = (int32_t)(total_length) - (int32_t)offset;
            bool is_valid_event_index =
                (index - 1) >= 0 && (index - 1) < (int32_t)(max_length);

            if (task->needs_exact() || !is_note) {
                if (task->add_event_data(
                        ((last_age / (60 * 24)) + p.birth_date).year(),
                        last_age, features, event.start_age_in_minutes,
                        is_valid_event_index, token_dropout != 0)) {
                    if (total_length == 0) {
                        throw std::runtime_error(
                            "Cannot create labels before birth " +
                            std::to_string(patient_offset) + " " +
                            std::to_string(last_age) + " " +
                            std::to_string(event.start_age_in_minutes));
                    }
                    if (is_valid_event_index) {
                        label_indices.push_back(batch_index * max_length +
                                                index - 1);
                    }
                }
            }

            if (index >= 0 && index < (int32_t)max_length && actually_add) {
                if ((batch_index * max_length + index) >= ages.dimension(0)) {
                    throw std::runtime_error("Really bad");
                }

                if (lookup.is_hierarchical) {
                    if (features.size() == 0) {
                        throw std::runtime_error(
                            "This should never happen ... ");
                    }
                    if (SPARSE_FALLBACK) {
                        bad_tokens(batch_index * max_length + index, 0) = 0;
                    }
                    for (uint32_t feature : features) {
                        if (feature >= lookup.vocab_size) {
                            throw std::runtime_error("Invalid feature ???");
                        }
                        sparse_token_indices_list.push_back(
                            {feature, batch_index * max_length + index});

                        if (SPARSE_FALLBACK) {
                            bad_tokens(batch_index * max_length + index,
                                       feature) = 1;
                        }
                    }
                } else {
                    tokens(batch_index * max_length + index) = features[0];
                }
                if (reader) {
                    is_note_embedding(batch_index * max_length + index) =
                        is_note;
                }
                valid_tokens(batch_index * max_length + index) = true;
                ages(batch_index * max_length + index) =
                    event.start_age_in_minutes / (60.0 * 24.0);
                normalized_ages(batch_index * max_length + index) =
                    (event.start_age_in_minutes / (60.0 * 24.0) - age_mean) /
                    (age_std);
            }

            total_length += 1;
            last_age = event.start_age_in_minutes;
        }

        int32_t index = (int32_t)(total_length) - (int32_t)offset;
        bool is_valid_event_index =
            (index - 1) >= 0 && (index - 1) < (int32_t)(max_length);
        if (task->add_event_data(((last_age / (60 * 24)) + p.birth_date).year(),
                                 last_age, {}, {}, is_valid_event_index,
                                 token_dropout != 0)) {
            if (total_length == 0) {
                throw std::runtime_error(
                    "Cannot create labels before birth (during "
                    ") for final patients" +
                    std::to_string(last_age) + " " +
                    std::to_string(p.patient_offset));
            }
            if (is_valid_event_index) {
                label_indices.push_back(batch_index * max_length + index - 1);
            }
        }

        current_note_offset += current_indices.size();
        batch_index++;
    }

    void prepare_batch_data() {
        uint32_t needed_reps = std::max(label_indices.size(), (size_t)256);

        uint32_t num_reps = round_to_nearest_bin(needed_reps, 2);

        label_indices_tensor = Eigen::Tensor<uint32_t, 1>(num_reps);
        label_indices_tensor.setConstant(ages.size());

        for (size_t i = 0; i < label_indices.size(); i++) {
            if (i >= 1) {
                if (label_indices[i] < label_indices[i - 1]) {
                    throw std::runtime_error(
                        absl::StrCat("Violated the order ", label_indices[i],
                                     label_indices[i - 1]));
                }
                if (label_indices[i] == label_indices[i - 1]) {
                    throw std::runtime_error(
                        absl::StrCat("Violated the unique ", label_indices[i],
                                     label_indices[i - 1]));
                }
            }
            label_indices_tensor(i) = label_indices[i];
        }

        if (lookup.is_hierarchical) {
            uint32_t num_sparse_tokens =
                round_to_nearest_bin(sparse_token_indices_list.size(), 2);
            sparse_token_indices =
                Eigen::Tensor<uint32_t, 2>(num_sparse_tokens, 2);

            for (uint32_t i = 0; i < num_sparse_tokens; i++) {
                if (i < sparse_token_indices_list.size()) {
                    sparse_token_indices(i, 0) =
                        sparse_token_indices_list[i][0];
                    sparse_token_indices(i, 1) =
                        sparse_token_indices_list[i][1];
                } else {
                    sparse_token_indices(i, 0) = lookup.vocab_size;
                    sparse_token_indices(i, 1) = ages.size();
                }
            }
        }

        if (reader) {
            size_t num_notes = current_note_embedding_bytes.size() / (2 * 768);

            uint32_t num_embed =
                round_to_nearest_bin(std::max((size_t)1, num_notes), 2);

            note_embedding_bytes =
                Eigen::Tensor<uint8_t, 1>(num_embed * 2 * 768);

            for (size_t i = 0; i < current_note_embedding_bytes.size(); i++) {
                note_embedding_bytes(i) = current_note_embedding_bytes[i];
            }
        }

        task->prepare_batch_data(num_reps);
    }

    const std::vector<uint32_t>& get_label_indices() const {
        return label_indices;
    }
    const Task* get_task() const { return task.get(); }

    py::dict get_batch() {
        py::dict transformer;
        transformer["length"] = max_length;
        if (lookup.is_hierarchical) {
            transformer["sparse_token_indices"] = sparse_token_indices;
            if (SPARSE_FALLBACK) {
                transformer["bad_tokens"] = bad_tokens;
            }
        } else {
            transformer["tokens"] = tokens;
        }
        if (reader) {
            transformer["is_note_embedding"] = is_note_embedding;
            transformer["note_embedding_bytes"] = note_embedding_bytes;
        }
        transformer["valid_tokens"] = valid_tokens;
        transformer["ages"] = ages;
        transformer["normalized_ages"] = normalized_ages;
        transformer["label_indices"] = label_indices_tensor;

        py::dict task_dict = task->get_batch_data();

        py::dict result;
        result["num_patients"] = batch_index;
        result["num_indices"] = label_indices.size();
        result["patient_ids"] = patient_ids;
        result["offsets"] = offsets;
        result["transformer"] = transformer;
        result["task"] = task_dict;
        return result;
    }

   private:
    PatientDatabase& data;
    PatientDatabaseIterator iter;
    FeatureLookup lookup;
    std::unique_ptr<Task> task;
    double age_mean;
    double age_std;
    SplitMix64 rng;
    double token_dropout;

    Eigen::Tensor<uint64_t, 1> patient_ids;
    Eigen::Tensor<uint32_t, 1> offsets;

    Eigen::Tensor<uint32_t, 1> tokens;
    Eigen::Tensor<float, 2> bad_tokens;
    Eigen::Tensor<bool, 1> valid_tokens;

    std::vector<std::array<uint32_t, 2>> sparse_token_indices_list;

    Eigen::Tensor<uint32_t, 2> sparse_token_indices;

    Eigen::Tensor<float, 1> ages;
    Eigen::Tensor<float, 1> normalized_ages;

    std::vector<uint32_t> label_indices;
    Eigen::Tensor<uint32_t, 1> label_indices_tensor;

    absl::flat_hash_set<uint32_t> codes_seen_today;

    uint32_t batch_index;
    uint32_t max_length;

    boost::optional<ConstdbReader> reader;
    std::vector<uint8_t> current_note_embedding_bytes;
    uint32_t current_note_offset;

    Eigen::Tensor<uint8_t, 1> note_embedding_bytes;
    Eigen::Tensor<bool, 1> is_note_embedding;
};

struct BatchInfo {
    std::map<std::string,
             std::vector<std::vector<std::pair<uint32_t, uint32_t>>>>
        binned_indices;

    boost::optional<BatchCreator> creator;
};

void add_patient_to_batch(
    BatchInfo& data, const Patient& p, uint32_t seed, uint32_t min_size,
    uint32_t max_size,
    const std::vector<std::tuple<std::string, uint32_t, uint32_t>>& splits,
    const json& config, PatientDatabase& dataset) {
    if (!data.creator) {
        for (const auto& split : splits) {
            data.binned_indices[std::get<0>(split)].resize(max_size + 1);
        }

        data.creator.emplace(dataset, config);
    }

    data.creator->start_batch(1 << 30);

    data.creator->add_patient(p.patient_offset, 0, false);

    const std::vector<uint32_t>& repr_indices =
        data.creator->get_label_indices();

    if (repr_indices.size() == 0) {
        return;
    }

    // First, try to do a basic prefix
    std::vector<uint32_t> start_indices;

    uint32_t max_length = 0;
    for (uint32_t val : repr_indices) {
        max_length = std::max(max_length, val);
    }

    if (data.creator->get_task()->needs_exact()) {
        for (uint32_t val : repr_indices) {
            if (val < ((uint32_t)1 << max_size)) {
                if (start_indices.size() == 0) {
                    start_indices.push_back(0);
                }
            } else {
                start_indices.push_back(val - ((uint32_t)1 << max_size) + 1);
            }
        }
    } else {
        if (max_length < ((uint32_t)1 << max_size)) {
            start_indices.push_back(0);
        } else {
            if (false && (repr_indices[0] >= ((uint32_t)1 << max_size))) {
                throw std::runtime_error("Could not work it? " +
                                         std::to_string(p.patient_offset) + " " +
                                         std::to_string(repr_indices[0]));
            }
            start_indices.push_back(max_length - ((uint32_t)1 << max_size) + 1);
        }
    }

    uint32_t bin_index = std::max(
        min_size, std::min(uint32_t(ceil(log2(max_length + 1))), max_size));

    uint32_t split_index = dataset.compute_split(seed, p.patient_offset);

    bool found_split = false;
    for (const auto& split : splits) {
        auto& name = std::get<0>(split);
        auto& start = std::get<1>(split);
        auto& end = std::get<2>(split);
        if (split_index >= start && split_index < end) {
            if (found_split) {
                std::cout << "You should not be getting duplicates ..."
                          << std::endl;
                abort();
            }
            found_split = true;
            for (uint32_t start_index : start_indices) {
                data.binned_indices[name][bin_index].push_back(
                    std::make_pair(p.patient_offset, start_index));
            }
        }
    }
}

void combine_batch_info(BatchInfo& data, const BatchInfo& source) {
    for (const auto& entry : source.binned_indices) {
        auto dest_iter = data.binned_indices.find(entry.first);
        if (dest_iter == std::end(data.binned_indices)) {
            data.binned_indices.insert(entry);
        } else {
            for (uint32_t bin_index = 0; bin_index < dest_iter->second.size();
                 bin_index++) {
                auto& dest_bin = dest_iter->second[bin_index];
                const auto& src_bin = entry.second[bin_index];

                dest_bin.insert(std::end(dest_bin), std::begin(src_bin),
                                std::end(src_bin));
            }
        }
    }
}

void create_batches(const std::string& target_path,
                    const std::string& path_to_data,
                    const std::string& batch_config_path) {
    PatientDatabase data(path_to_data, true);
    data.get_ontology().get_all_parents(0);
    data.get_ontology().get_dictionary().size();
    data.compute_split(0, 0);
    data.get_patient(0);

    json config = read_file(batch_config_path);

    json result;
    result["config"] = config;
    result["data_path"] = path_to_data;

    uint32_t min_size = config["transformer"]["min_size"];
    uint32_t max_size = config["transformer"]["max_size"];
    uint32_t seed = config["seed"];

    std::mt19937 g(seed);

    std::vector<std::tuple<std::string, uint32_t, uint32_t>> splits =
        config["splits"];

    absl::flat_hash_set<uint32_t> valid_patients;

    auto task = create_task(config["task"], data, data.get_ontology());

    if (task->get_patient_offsets().size() != 0) {
        for (uint32_t patient_offset : task->get_patient_offsets()) {
            valid_patients.insert(patient_offset);
        }
    }

    (void)data.get_unique_text_dictionary();
    (void)data.get_shared_text_dictionary();
    (void)data.get_shared_text_dictionary().init_sorted_values();
    (void)data.get_ontology().get_dictionary();
    (void)data.get_ontology().get_dictionary().init_sorted_values();

    BatchInfo info = proccess_patients_in_parallel(
        data, 40,
        [&](BatchInfo& info, const Patient& p) {
            if (valid_patients.size() > 0 &&
                valid_patients.count(p.patient_offset) == 0) {
                return;
            }
            add_patient_to_batch(info, p, seed, min_size, max_size, splits,
                                 config, data);
        },
        combine_batch_info);

    std::map<std::string,
             std::vector<std::pair<uint32_t,
                                   std::vector<std::pair<uint32_t, uint32_t>>>>>
        batches;

    for (const auto& split : splits) {
        const auto& split_name = std::get<0>(split);
        auto& bins = info.binned_indices[split_name];

        std::vector<
            std::pair<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>>>
            res_batches;

        for (uint32_t size = 0; size <= max_size; size++) {
            auto& bin = bins[size];
            std::shuffle(std::begin(bin), std::end(bin), g);
            uint32_t num_per_batch = 1 << (max_size - size);
            std::vector<std::pair<uint32_t, uint32_t>> next_batch;
            for (const auto& entry : bin) {
                if (next_batch.size() == num_per_batch) {
                    res_batches.push_back(std::make_pair(size, next_batch));
                    next_batch.clear();
                }
                next_batch.push_back(entry);
            }
            if (next_batch.size() > 0) {
                res_batches.push_back(std::make_pair(size, next_batch));
            }
        }

        std::shuffle(std::begin(res_batches), std::end(res_batches), g);

        batches[split_name] = res_batches;
    }

    result["batches"] = batches;

    std::ofstream output(target_path, std::ios::binary);
    json::to_msgpack(result, output);
}

class BatchLoader {
   public:
    BatchLoader(std::string path_to_data, std::string batch_info_path,
                double _token_dropout = 0)
        : data(path_to_data, false),
          batch_info(read_file(batch_info_path)),
          config(batch_info["config"]),
          batch_creator(data, config, _token_dropout),
          batches(batch_info["batches"]) {}

    size_t get_number_of_batches(const std::string& split) {
        return batches[split].size();
    }
    py::dict get_batch(const std::string& split, uint32_t index) {
        {
            py::gil_scoped_release release;

            auto batch_iter = batches.find(split);
            if (batch_iter == std::end(batches)) {
                throw std::runtime_error("Could not find batches for split ? " +
                                         split);
            }

            const auto& batch_vector = batch_iter->second;

            if (index >= batch_vector.size()) {
                throw std::runtime_error("Batch index is larger than batch " +
                                         split + " " + std::to_string(index) +
                                         " " + std::to_string(batches.size()));
            }

            const auto& batch = batch_vector[index];

            uint32_t max_size = config["transformer"]["max_size"];
            uint32_t size = batch.first;
            uint32_t num_per_batch = 1 << (max_size - size);
            uint32_t batch_size = 1 << size;
            batch_creator.start_batch(batch_size);

            if (batch.second.size() > num_per_batch) {
                throw std::runtime_error("Too many tokens");
            }

            for (const auto& entry : batch.second) {
                batch_creator.add_patient(entry.first, entry.second);
            }

            batch_creator.prepare_batch_data();
        }

        return batch_creator.get_batch();
    }

   private:
    PatientDatabase data;
    json batch_info;
    json config;
    BatchCreator batch_creator;
    std::map<std::string,
             std::vector<std::pair<uint32_t,
                                   std::vector<std::pair<uint32_t, uint32_t>>>>>
        batches;
};

void register_dataloader_extension(pybind11::module& root) {
    py::module m = root.def_submodule("dataloader");

    py::class_<BatchLoader>(m, "BatchLoader")
        .def(py::init<std::string, std::string, double>(),
             py::arg("path_to_data"), py::arg("batch_info_path"),
             py::arg("token_dropout") = 0)
        .def("get_number_of_batches", &BatchLoader::get_number_of_batches)
        .def("get_batch", &BatchLoader::get_batch);

    m.def("create_batches", create_batches);
    m.def("create_dictionary", create_dictionary);
    m.def("create_survival_dictionary", create_survival_dictionary);
}
