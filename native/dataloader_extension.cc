#include "dataloader_extension.hh"

#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace py = pybind11;

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <queue>
#include <random>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "clmbr_dictionary.hh"
#include "database.hh"
#include "flatmap.hh"
#include "pybind11/eigen/tensor.h"
#include "survival.hh"

class Task {
   public:
    virtual ~Task(){};

    virtual const std::vector<uint32_t>& get_patient_ids() = 0;
    virtual void start_batch() = 0;
    virtual void start_patient(const Patient& p) = 0;

    virtual bool add_event_data(int current_year, float current_age,
                                uint32_t next_feature, float next_age) = 0;

    virtual void prepare_batch_data(uint32_t num_representations) = 0;
    virtual py::dict get_batch_data() = 0;
};

uint32_t round_to_nearest_bin(uint32_t value, uint32_t skip = 1) {
    uint32_t result = 1;
    while (value > result) {
        result <<= skip;
    }
    assert(result >= value);
    return result;
}

class BinaryTask : public Task {
   public:
    BinaryTask(json config) {
        for (json label : config["labels"]) {
            uint32_t patient_id = label[0];
            float age = label[1];
            bool value = label[2];
            labels[patient_id].push_back(std::make_pair(age, value));
        }

        for (auto& entry : labels) {
            patient_ids.push_back(entry.first);
            std::sort(std::begin(entry.second), std::end(entry.second));
        }
    }

    const std::vector<uint32_t>& get_patient_ids() override {
        return patient_ids;
    }
    void start_batch() override { batch_labels.clear(); }

    void start_patient(const Patient& p) override {
        current_patient_labels = &(labels.find(p.patient_id)->second);
        current_label_iter = std::begin(*current_patient_labels);
    }

    bool add_event_data(int current_year, float current_age,
                        uint32_t next_feature, float next_age) override {
        bool added_one = false;
        while (current_label_iter != std::end(*current_patient_labels) &&
               current_label_iter->first < next_age) {
            if (added_one) {
                throw std::runtime_error(
                    "Currently only supports one label per event ...");
            }
            batch_labels.push_back(current_label_iter->second);
            added_one = true;

            current_label_iter++;
        }

        return added_one;
    }

    Eigen::Tensor<float, 1> final_batch_labels;
    virtual void prepare_batch_data(uint32_t num_representations) override {
        final_batch_labels = Eigen::Tensor<float, 1>(num_representations);
        final_batch_labels.setConstant(0);

        for (uint32_t i = 0; i < batch_labels.size(); i++) {
            final_batch_labels(i) = batch_labels[i];
        }
    }

    py::dict get_batch_data() override {
        py::dict result;
        result["labels"] = final_batch_labels;
        return result;
    }

   private:
    absl::flat_hash_map<uint32_t, std::vector<std::pair<float, bool>>> labels;
    std::vector<uint32_t> patient_ids;

    std::vector<float> batch_labels;

    const std::vector<std::pair<float, bool>>* current_patient_labels;
    std::vector<std::pair<float, bool>>::const_iterator current_label_iter;
};

class SurvivalTask : public Task {
   public:
    SurvivalTask(json config) {
        for (json label : config["labels"]) {
            uint32_t patient_id = label[0];
            float age = label[1];
            float time = label[2];
            bool is_censor = label[3];
            labels[patient_id].push_back(
                std::make_pair(age, std::make_pair(time, is_censor)));
        }

        for (auto& entry : labels) {
            patient_ids.push_back(entry.first);
            std::sort(std::begin(entry.second), std::end(entry.second));
        }
    }

    const std::vector<uint32_t>& get_patient_ids() override {
        return patient_ids;
    }
    void start_batch() override { batch_data.clear(); }

    void start_patient(const Patient& p) override {
        current_patient_labels = &(labels.find(p.patient_id)->second);
        current_label_iter = std::begin(*current_patient_labels);
    }

    bool add_event_data(int current_year, float current_age,
                        uint32_t next_feature, float next_age) override {
        bool added_one = false;
        while (current_label_iter != std::end(*current_patient_labels) &&
               current_label_iter->first < next_age) {
            if (added_one) {
                throw std::runtime_error(
                    "Currently only supports one label per event ...");
            }
            auto data = current_label_iter->second;
            data.first -= current_age;
            batch_data.push_back(data);
            added_one = true;

            current_label_iter++;
        }

        return added_one;
    }

    Eigen::Tensor<float, 1> final_batch_times;
    Eigen::Tensor<float, 1> final_batch_is_censor;

    virtual void prepare_batch_data(uint32_t num_representations) override {
        final_batch_times = Eigen::Tensor<float, 1>(num_representations);
        final_batch_is_censor = Eigen::Tensor<float, 1>(num_representations);
        final_batch_times.setConstant(0);
        final_batch_is_censor.setConstant(1);

        for (uint32_t i = 0; i < batch_data.size(); i++) {
            final_batch_times(i) = batch_data[i].first;
            final_batch_is_censor(i) = batch_data[i].second;
        }
    }

    py::dict get_batch_data() override {
        py::dict result;
        result["times"] = final_batch_times;
        result["is_censor"] = final_batch_is_censor;

        return result;
    }

   private:
    absl::flat_hash_map<uint32_t,
                        std::vector<std::pair<float, std::pair<float, bool>>>>
        labels;
    std::vector<uint32_t> patient_ids;

    std::vector<std::pair<float, bool>> batch_data;

    const std::vector<std::pair<float, std::pair<float, bool>>>*
        current_patient_labels;
    std::vector<std::pair<float, std::pair<float, bool>>>::const_iterator
        current_label_iter;
};

class CLMBRTask : public Task {
   public:
    CLMBRTask(json config) {
        // Might be empty, in which case we train on everyone
        patient_ids = config.value("patient_ids", std::vector<uint32_t>());
        vocab_size = config["vocab_size"];
    }

    const std::vector<uint32_t>& get_patient_ids() override {
        return patient_ids;
    }

    void start_batch() override { batch_labels.clear(); }

    void start_patient(const Patient& p) override {}

    bool add_event_data(int current_year, float current_age,
                        uint32_t next_feature, float next_age) override {
        if (next_feature >= vocab_size) {
            return false;
        }
        if (next_age < 2 || isinf(next_age)) {
            // Don't try to predict stuff on the day of birth
            return false;
        }
        batch_labels.push_back(next_feature);
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

    py::dict get_batch_data() override {
        py::dict result;
        result["labels"] = final_batch_labels;

        return result;
    }

   private:
    uint32_t vocab_size;

    std::vector<uint32_t> patient_ids;

    std::vector<uint32_t> batch_labels;
};

class SurvivalCLMBRTask : public Task {
   public:
    SurvivalCLMBRTask(json config, Ontology& ontology) {
        // Might be empty, in which case we train on everyone
        patient_ids = config.value("patient_ids", std::vector<uint32_t>());
        json survival_dict = config["survival_dict"];
        survival_dictionary = get_mapping(ontology, survival_dict["codes"]);
        vocab_size = survival_dict["codes"].size();
        time_bins = survival_dict["time_bins"].get<std::vector<float>>();
    }

    const std::vector<uint32_t>& get_patient_ids() override {
        return patient_ids;
    }

    void start_batch() override {
        events_per_rep.clear();
        total_events = 0;
    }

    void start_patient(const Patient& p) override {
        calculator.preprocess_patient(p, survival_dictionary);
        last_prediction_age = 0;
    }

    bool add_event_data(int current_year, float current_age,
                        uint32_t next_feature, float next_age) override {
        if (!should_make_prediction(last_prediction_age, current_age, next_age,
                                    current_year)) {
            return false;
        }

        auto entry = calculator.get_times_for_event(current_age);
        if (entry.second.size() == 0) {
            return false;
        }

        last_prediction_age = current_age;

        total_events += entry.second.size();
        events_per_rep.push_back(entry);

        return true;
    }

    Eigen::Tensor<uint32_t, 2> batch_event_indices;
    Eigen::Tensor<uint32_t, 1> batch_event_offsets;
    Eigen::Tensor<float, 1> batch_censor_log_times;

    Eigen::Tensor<uint32_t, 1> batch_event_codes;
    Eigen::Tensor<float, 1> batch_event_log_times;
    void prepare_batch_data(uint32_t num_representations) override {
        uint32_t num_batch_events = round_to_nearest_bin(total_events, 4);

        batch_event_indices = Eigen::Tensor<uint32_t, 2>(num_batch_events, 2);
        for (uint32_t i = 0; i < num_batch_events; i++) {
            batch_event_indices(i, 0) = num_representations * time_bins.size();
            batch_event_indices(i, 1) = vocab_size;
        }

        batch_event_offsets = Eigen::Tensor<uint32_t, 1>(
            num_representations * time_bins.size() + 1);
        batch_event_offsets.setConstant(num_batch_events);

        batch_censor_log_times =
            Eigen::Tensor<float, 1>(num_representations * time_bins.size());
        batch_censor_log_times.setConstant(
            -std::numeric_limits<float>::infinity());

        batch_event_codes = Eigen::Tensor<uint32_t, 1>(num_batch_events);
        batch_event_codes.setConstant(vocab_size);

        batch_event_log_times = Eigen::Tensor<float, 1>(num_batch_events);
        batch_event_log_times.setConstant(
            std::numeric_limits<float>::quiet_NaN());

        uint32_t event_index = 0;
        for (uint32_t rep = 0; rep < events_per_rep.size(); rep++) {
            auto& entry = events_per_rep[rep];
            float censor = entry.first;
            auto& events = entry.second;
            std::sort(std::begin(events), std::end(events));

            auto event_iter = std::begin(events);

            for (uint32_t time_bin = 0; time_bin < time_bins.size();
                 time_bin++) {
                float start = time_bins[time_bin];
                float end;
                if (time_bin == time_bins.size()) {
                    end = std::numeric_limits<float>::infinity();
                } else {
                    end = time_bins[time_bin + 1];
                }

                float time_in_bin = censor - start;
                time_in_bin = std::min(time_in_bin, end - start);

                if (time_in_bin > 0) {
                    batch_censor_log_times(rep * time_bins.size() + time_bin) =
                        std::log2(time_in_bin);
                }

                batch_event_offsets(rep * time_bins.size() + time_bin) =
                    event_index;

                std::vector<std::pair<float, uint32_t>> events_for_bin;

                while (event_iter != std::end(events) &&
                       event_iter->first < end) {
                    events_for_bin.push_back(*event_iter++);
                }
                std::sort(std::begin(events_for_bin), std::end(events_for_bin),
                          [&](const auto& a, const auto& b) {
                              return a.second < b.second;
                          });

                for (const auto& event : events_for_bin) {
                    batch_event_indices(event_index, 0) =
                        rep * time_bins.size() + time_bin;
                    batch_event_indices(event_index, 1) = event.second;

                    float log_time = std::log2(event.first - start);
                    batch_event_codes(event_index) = event.second;
                    batch_event_log_times(event_index) = log_time;

                    event_index++;
                }
            }
        }

        assert(event_index == total_events);
    }

    py::dict get_batch_data() override {
        py::list sparse_time;
        sparse_time.append(batch_event_offsets);
        sparse_time.append(batch_censor_log_times);
        sparse_time.append(batch_event_codes);
        sparse_time.append(batch_event_log_times);

        py::dict result;
        result["num_valid"] = total_events;
        result["event_indices"] = batch_event_indices;
        result["sparse_time"] = sparse_time;

        return result;
    }

   private:
    std::vector<std::pair<float, std::vector<std::pair<float, uint32_t>>>>
        events_per_rep;
    size_t total_events;

    float last_prediction_age;
    SurvivalCalculator calculator;

    std::vector<uint32_t> patient_ids;

    FlatMap<std::vector<uint32_t>> survival_dictionary;
    uint32_t vocab_size;
    std::vector<float> time_bins;
};

std::unique_ptr<Task> create_task(json config, Ontology& ontology) {
    std::string type = config["type"];
    if (type == "binary") {
        return std::make_unique<BinaryTask>(config);
    } else if (type == "clmbr") {
        return std::make_unique<CLMBRTask>(config);
    } else if (type == "survival_clmbr") {
        return std::make_unique<SurvivalCLMBRTask>(config, ontology);
    }

    throw std::runtime_error("Invalid task type " + type);
}

class FeatureLookup {
   public:
    FeatureLookup(json data, uint32_t vocab_size) {
        json actual_data = data["regular"];

        for (uint32_t i = 0; i < vocab_size; i++) {
            DictEntry entry = actual_data[i].get<DictEntry>();
            switch (entry.type) {
                case DictEntryType::CODE:
                    code_features[entry.code] = i;
                    break;

                case DictEntryType::TEXT:
                    text_features[std::make_pair(entry.code,
                                                 entry.text_value)] = i;
                    break;

                case DictEntryType::NUMERIC:
                    numeric_features[entry.code].push_back(
                        std::make_tuple(entry.val_start, entry.val_end, i));
                    break;
            }
        }
    }

    boost::optional<uint32_t> get_feature_code(const Event& event) const {
        switch (event.value_type) {
            case ValueType::NONE: {
                auto iter = code_features.find(event.code);
                if (iter == std::end(code_features)) {
                    return boost::none;
                }
                return iter->second;
            }

            case ValueType::NUMERIC: {
                auto iter = numeric_features.find(event.code);
                if (iter == std::end(numeric_features)) {
                    return boost::none;
                }
                for (const auto& item : iter->second) {
                    if (event.numeric_value >= std::get<0>(item) &&
                        event.numeric_value < std::get<1>(item)) {
                        return std::get<2>(item);
                    }
                }
                return boost::none;
            }

            case ValueType::SHARED_TEXT: {
                auto iter = text_features.find(
                    std::make_pair(event.code, event.text_value));
                if (iter == std::end(text_features)) {
                    return boost::none;
                }
                return iter->second;
            }

            case ValueType::UNIQUE_TEXT:
                return boost::none;

            default:
                throw std::runtime_error("Invalid value type?");
        }
    }

   private:
    absl::flat_hash_map<std::pair<uint32_t, uint32_t>, uint32_t> text_features;
    absl::flat_hash_map<uint32_t,
                        std::vector<std::tuple<float, float, uint32_t>>>
        numeric_features;
    absl::flat_hash_map<uint32_t, uint32_t> code_features;
};

json read_file(std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    return json::from_msgpack(file);
}

struct BatchInfo {
    std::map<std::string, std::vector<std::vector<uint32_t>>> binned_indices;

    absl::flat_hash_set<uint32_t> codes_seen_today;

    std::unique_ptr<Task> task;
};

void add_patient_to_batch(
    BatchInfo& data, const Patient& p, uint32_t seed, uint32_t min_size,
    uint32_t max_size,
    const std::vector<std::tuple<std::string, uint32_t, uint32_t>>& splits,
    const json& config, const FeatureLookup& lookup, PatientDatabase& dataset) {
    if (!data.task) {
        for (const auto& split : splits) {
            data.binned_indices[std::get<0>(split)].resize(max_size + 1);
        }

        data.task = create_task(config["task"], dataset.get_ontology());
    }

    data.codes_seen_today.clear();

    float last_age = 0;

    uint32_t total_length = 0;
    uint32_t max_length = 0;

    data.task->start_batch();

    data.task->start_patient(p);

    for (const auto& event : p.events) {
        if ((int)event.age != (int)last_age) {
            data.codes_seen_today.clear();
        }

        auto feature = lookup.get_feature_code(event);
        if (!feature) {
            continue;
        }

        if (data.codes_seen_today.count(*feature) > 0) {
            continue;
        }

        data.codes_seen_today.insert(*feature);

        if (total_length == (((uint32_t)1) << max_size)) {
            break;
        }

        if (data.task->add_event_data((p.birth_date + last_age).year(),
                                      last_age, *feature, event.age)) {
            if (total_length == 0) {
                throw std::runtime_error(
                    "Cannot create labels before birth (during "
                    "preprocessing)");
            }
            max_length = std::max(max_length, total_length - 1);
        }

        total_length++;
        last_age = event.age;
    }

    if (data.task->add_event_data((p.birth_date + last_age).year(), last_age, 0,
                                  std::numeric_limits<float>::infinity())) {
        max_length = std::max(max_length, total_length - 1);
    }

    if (max_length == 0) {
        return;
    }

    uint32_t bin_index = std::max(
        min_size, std::min(uint32_t(ceil(log2(max_length + 1))), max_size));

    uint32_t split_index = dataset.compute_split(seed, p.patient_id);

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
            data.binned_indices[name][bin_index].push_back(p.patient_id);
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

    json config = read_file(batch_config_path);

    FeatureLookup lookup(config["transformer"]["dictionary"],
                         config["transformer"]["vocab_size"]);
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

    auto task = create_task(config["task"], data.get_ontology());

    if (task->get_patient_ids().size() != 0) {
        for (uint32_t patient_id : task->get_patient_ids()) {
            valid_patients.insert(patient_id);
        }
    }

    BatchInfo info = proccess_patients_in_parallel(
        data, 40,
        [&](BatchInfo& info, const Patient& p) {
            if (valid_patients.size() > 0 &&
                valid_patients.count(p.patient_id) == 0) {
                return;
            }
            add_patient_to_batch(info, p, seed, min_size, max_size, splits,
                                 config, lookup, data);
        },
        combine_batch_info);

    std::map<std::string,
             std::vector<std::pair<uint32_t, std::vector<uint32_t>>>>
        batches;

    for (const auto& split : splits) {
        const auto& split_name = std::get<0>(split);
        auto& bins = info.binned_indices[split_name];

        std::vector<std::pair<uint32_t, std::vector<uint32_t>>> res_batches;

        for (uint32_t size = 0; size <= max_size; size++) {
            auto& bin = bins[size];
            std::shuffle(std::begin(bin), std::end(bin), g);
            uint32_t num_per_batch = 1 << (max_size - size);
            for (uint32_t i = 0; i < bin.size(); i += num_per_batch) {
                std::vector<uint32_t> next_batch;
                for (uint32_t j = i; j < bin.size() && j < i + num_per_batch;
                     j++) {
                    next_batch.push_back(bin[j]);
                }
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

class BatchCreator {
   public:
    BatchCreator(std::string path_to_data, std::string batch_info_path)
        : data(path_to_data, true),
          iter(data.iterator()),
          batch_info(read_file(batch_info_path)),
          config(batch_info["config"]),
          lookup(config["transformer"]["dictionary"],
                 config["transformer"]["vocab_size"]),
          task(create_task(config["task"], data.get_ontology())) {
        uint32_t min_size = config["transformer"]["min_size"];
        uint32_t max_size = config["transformer"]["max_size"];

        age_mean = config["transformer"]["dictionary"]["age_stats"]["mean"];
        age_std = config["transformer"]["dictionary"]["age_stats"]["std"];

        batches = batch_info["batches"];

        patient_ids = Eigen::Tensor<uint32_t, 1>(1 << (max_size - min_size));
        tokens = Eigen::Tensor<uint32_t, 1>(1 << max_size);
        ages = Eigen::Tensor<float, 1>(1 << max_size);
        normalized_ages = Eigen::Tensor<float, 1>(1 << max_size);
    }

    size_t get_number_of_batches(const std::string& split) {
        return batches[split].size();
    }

    Eigen::Tensor<uint32_t, 1> patient_ids;
    Eigen::Tensor<uint32_t, 1> tokens;
    Eigen::Tensor<float, 1> ages;
    Eigen::Tensor<float, 1> normalized_ages;
    std::vector<uint32_t> label_indices;
    absl::flat_hash_set<uint32_t> codes_seen_today;
    Eigen::Tensor<uint32_t, 1> label_indices_tensor;

    py::dict get_batch(const std::string& split, size_t index) {
        py::dict result;
        const auto& batch = batches[split][index];

        uint32_t max_size = config["transformer"]["max_size"];
        uint32_t size = batch.first;
        uint32_t num_per_batch = 1 << (max_size - size);
        uint32_t batch_size = 1 << size;

        {
            py::gil_scoped_release release;

            patient_ids.setConstant(0);
            tokens.setConstant(0);
            ages.setConstant(0);
            normalized_ages.setConstant(0);

            label_indices.clear();
            codes_seen_today.clear();

            task->start_batch();

            for (size_t i = 0; i < batch.second.size(); i++) {
                uint32_t patient_id = batch.second[i];
                patient_ids(i) = patient_id;

                const Patient& p = iter.get_patient(patient_id);

                task->start_patient(p);

                codes_seen_today.clear();
                float last_age = 0;

                uint32_t total_length = 0;

                for (const auto& event : p.events) {
                    if ((int)event.age != (int)last_age) {
                        codes_seen_today.clear();
                    }

                    auto feature = lookup.get_feature_code(event);
                    if (!feature) {
                        continue;
                    }

                    if (codes_seen_today.count(*feature) > 0) {
                        continue;
                    }
                    codes_seen_today.insert(*feature);

                    if (total_length == batch_size) {
                        break;
                    }

                    if (task->add_event_data((last_age + p.birth_date).year(),
                                             last_age, *feature, event.age)) {
                        if (total_length == 0) {
                            throw std::runtime_error(
                                "Cannot create labels before birth " +
                                std::to_string(patient_id));
                        }
                        label_indices.push_back(i * batch_size + total_length -
                                                1);
                    }

                    tokens(i * batch_size + total_length) = *feature;
                    ages(i * batch_size + total_length) = event.age;
                    normalized_ages(i * batch_size + total_length) =
                        (event.age - age_mean) / (age_std);

                    total_length += 1;
                    last_age = event.age;
                }

                if (task->add_event_data(
                        (last_age + p.birth_date).year(), last_age, 0,
                        std::numeric_limits<float>::infinity())) {
                    label_indices.push_back(i * batch_size + total_length - 1);
                }
            }

            uint32_t needed_reps = std::max(label_indices.size(), (size_t)256);

            uint32_t num_reps = round_to_nearest_bin(needed_reps, 2);

            label_indices_tensor = Eigen::Tensor<uint32_t, 1>(num_reps);
            label_indices_tensor.setConstant(batch_size * num_per_batch);

            for (size_t i = 0; i < label_indices.size(); i++) {
                label_indices_tensor(i) = label_indices[i];
            }

            task->prepare_batch_data(num_reps);
        }

        py::dict transformer;
        transformer["length"] = 1 << size;
        transformer["tokens"] = tokens;
        transformer["ages"] = ages;
        transformer["normalized_ages"] = normalized_ages;
        transformer["label_indices"] = label_indices_tensor;

        py::dict task_dict = task->get_batch_data();

        result["num_patients"] = 1 << (max_size - size);
        result["patient_ids"] = patient_ids;
        result["transformer"] = transformer;
        result["task"] = task_dict;
        return result;
    }

   private:
    PatientDatabase data;
    PatientDatabaseIterator iter;
    json batch_info;
    json config;
    FeatureLookup lookup;
    std::map<std::string,
             std::vector<std::pair<uint32_t, std::vector<uint32_t>>>>
        batches;
    std::unique_ptr<Task> task;
    double age_mean;
    double age_std;
};

void register_dataloader_extension(pybind11::module& root) {
    py::module m = root.def_submodule("dataloader");

    py::class_<BatchCreator>(m, "BatchCreator")
        .def(py::init<std::string, std::string>())
        .def("get_number_of_batches", &BatchCreator::get_number_of_batches)
        .def("get_batch", &BatchCreator::get_batch);

    m.def("create_batches", create_batches);
}
