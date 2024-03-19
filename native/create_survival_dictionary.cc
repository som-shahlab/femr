#include "create_survival_dictionary.hh"

#include <iostream>
#include <nlohmann/json.hpp>
#include <random>

#include "absl/container/flat_hash_map.h"
#include "database.hh"
#include "flatmap.hh"
#include "stat_utils.hh"
#include "survival.hh"

using json = nlohmann::json;

struct SurvivalDictionaryData {
    FlatMap<double> true_counts;

    SurvivalCalculator calculator;
};

void combine_data(SurvivalDictionaryData& target,
                  const SurvivalDictionaryData& source) {
    for (uint32_t code : source.true_counts.keys()) {
        *target.true_counts.find_or_insert(code, 0) +=
            *source.true_counts.find(code);
    }
}

void process_patient(SurvivalDictionaryData& data, const Patient& p,
                     Ontology& ontology, size_t num_patients,
                     const FlatMap<bool>& banned_codes) {
    data.calculator.preprocess_patient(p, [&](uint32_t code) {
        if (banned_codes.find(code) != nullptr) {
            return absl::Span<const uint32_t>();
        } else {
            return ontology.get_all_parents(code);
        }
    });

    double weight = 1.0 / (p.events.size() * num_patients);

    uint32_t last_prediction_age = 0;
    for (size_t event_index = 0; event_index < (p.events.size() - 1);
         event_index++) {
        const Event& event = p.events[event_index];
        const Event& next_event = p.events[event_index + 1];

        if (!should_make_prediction(
                last_prediction_age, event.start_age_in_minutes,
                next_event.start_age_in_minutes,
                (p.birth_date + event.start_age_in_minutes / (60 * 24))
                    .year())) {
            continue;
        }

        auto entry =
            data.calculator.get_times_for_event(event.start_age_in_minutes)
                .second;

        if (entry.size() == 0) {
            continue;
        }

        last_prediction_age = event.start_age_in_minutes;

        for (const auto& event : entry) {
            *data.true_counts.find_or_insert(event.second, 0) += weight;
        }
    }
}

struct TimeBinCollectionData {
    SurvivalCalculator calculator;

    FlatMap<ReservoirSampler<uint32_t>> sample_per_index;

    FlatMap<std::tuple<OnlineStatistics, double, double>> times_per_index;

    std::mt19937 rng;
};

void combine_time_data(TimeBinCollectionData& target,
                       const TimeBinCollectionData& source) {
    for (uint32_t code : source.sample_per_index.keys()) {
        auto source_sampler = source.sample_per_index.find(code);
        auto sampler = target.sample_per_index.find_or_insert(
            code, ReservoirSampler<uint32_t>(10000));
        sampler->combine(*source_sampler, target.rng);
    }

    for (uint32_t code : source.times_per_index.keys()) {
        auto source_times = source.times_per_index.find(code);
        auto times = target.times_per_index.find_or_insert(
            code, std::make_tuple(OnlineStatistics(), 0, 0));
        std::get<0>(*times).combine(std::get<0>(*source_times));
        std::get<1>(*times) += std::get<1>(*source_times);
        std::get<2>(*times) += std::get<2>(*source_times);
    }
}

void process_time_patient(TimeBinCollectionData& data, const Patient& p,
                          size_t num_patients,
                          const FlatMap<std::vector<uint32_t>>& indices,
                          const std::vector<uint32_t>& codes) {
    data.calculator.preprocess_patient(p, indices);

    double weight = 1.0 / (p.events.size() * num_patients);

    uint32_t last_prediction_age = 0;

    for (size_t event_index = 0; event_index < (p.events.size() - 1);
         event_index++) {
        const Event& event = p.events[event_index];
        const Event& next_event = p.events[event_index + 1];

        if (!should_make_prediction(
                last_prediction_age, event.start_age_in_minutes,
                next_event.start_age_in_minutes,
                (p.birth_date + event.start_age_in_minutes / (60 * 24))
                    .year())) {
            continue;
        }

        auto full =
            data.calculator.get_times_for_event(event.start_age_in_minutes);
        auto entry = full.second;
        if (entry.size() > 0) {
            for (const auto& event : entry) {
                data.sample_per_index
                    .find_or_insert(event.second,
                                    ReservoirSampler<uint32_t>(10000))
                    ->add(event.first, weight, data.rng);
            }
            last_prediction_age = event.start_age_in_minutes;
        }
        std::sort(
            std::begin(entry), std::end(entry),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        auto entry_iter = std::begin(entry);
        for (uint32_t i = 0; i < codes.size(); i++) {
            auto times = data.times_per_index.find_or_insert(
                i, std::make_tuple(OnlineStatistics(), 0, 0));

            double w = 1;

            if (entry_iter != std::end(entry) && entry_iter->second == i) {
                std::get<0>(*times).add_value(w, entry_iter->first);
                std::get<1>(*times) += w;
                entry_iter++;
            } else {
                std::get<0>(*times).add_value(w, full.first);
                std::get<2>(*times) += w;
            }
        }
    }
}

void create_survival_dictionary(const std::string& input,
                                const std::string& output, size_t num_buckets,
                                size_t dictionary_size) {
    boost::filesystem::path path(input);
    PatientDatabase database(path, true);

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

    // Prime the pump
    database.get_ontology().get_all_parents(0);

    SurvivalDictionaryData result = proccess_patients_in_parallel(
        database, 32,
        [&](SurvivalDictionaryData& res, const Patient& p) {
            process_patient(res, p, database.get_ontology(), database.size(),
                            banned_codes);
        },
        combine_data);

    std::vector<std::pair<double, uint32_t>> options;

    for (uint32_t code : result.true_counts.keys()) {
        double weight = *result.true_counts.find(code);
        double baseline = 1;

        for (uint32_t parent : database.get_ontology().get_parents(code)) {
            baseline = std::min(*result.true_counts.find(parent), baseline);
        }

        weight = weight / baseline;

        double info = 0;
        if (weight != 0 && weight != 1) {
            info = baseline *
                   (weight * log(weight) + (1 - weight) * log(1 - weight));
        }

        options.push_back(std::make_pair(info, code));
    }

    std::sort(std::begin(options), std::end(options));

    FlatMap<bool> used;

    std::vector<uint32_t> result_options;

    std::cout << "Starting to process " << std::endl;

    for (const auto& entry : options) {
        result_options.push_back(entry.second);
    }

    result_options.resize(dictionary_size);

    auto d = get_mapping(database.get_ontology(), result_options);

    std::mt19937 rng(12363);

    TimeBinCollectionData time_bin_result = proccess_patients_in_parallel(
        database, 32,
        [&](TimeBinCollectionData& res, const Patient& p) {
            process_time_patient(res, p, database.size(), d, result_options);
        },
        combine_time_data);

    auto get_buckets = [&](const ReservoirSampler<uint32_t>& samples,
                           size_t n_b) {
        std::vector<uint32_t> result(n_b);

        auto s = samples.get_samples();
        std::sort(std::begin(s), std::end(s));

        result[0] = 0;
        for (size_t i = 1; i < n_b; i++) {
            size_t index = ((double)i / (double)(n_b)) * (s.size() - 1);
            result[i] = s[index];
        }
        return result;
    };

    auto print_buckets = [&](const ReservoirSampler<uint32_t>& samples,
                             size_t n_b) {
        auto s = samples.get_samples();
        std::cout << "Got total weight " << samples.get_total_weight()
                  << std::endl;

        auto buckets = get_buckets(samples, n_b);

        for (auto val : buckets) {
            std::cout << val << " ";
        }

        std::cout << std::endl;
    };

    ReservoirSampler<uint32_t> global_samples(10000);
    ReservoirSampler<uint32_t> reweighted_global_samples(10000);

    for (uint32_t i = 0; i < dictionary_size; i++) {
        global_samples.combine(*time_bin_result.sample_per_index.find(i), rng);
        reweighted_global_samples.combine(
            *time_bin_result.sample_per_index.find(i), rng, 1);
    }

    std::cout << "RAEDY" << std::endl;
    print_buckets(global_samples, 8);
    print_buckets(reweighted_global_samples, 8);
    print_buckets(global_samples, 16);
    print_buckets(reweighted_global_samples, 16);

    std::vector<double> lambdas;

    for (uint32_t i = 0; i < dictionary_size; i++) {
        auto stats = time_bin_result.times_per_index.find(i);

        double frac_events =
            std::get<1>(*stats) / (std::get<1>(*stats) + std::get<2>(*stats));
        double lambda = frac_events / std::get<0>(*stats).get_mean();

        // std::cout << i << " Num total "
        //           << (std::get<1>(*stats) + std::get<2>(*stats))
        //           << " Num events " << std::get<1>(*stats) << " num censored"
        //           << std::get<2>(*stats) << std::endl;
        // std::cout << "Mean time " << std::get<0>(*stats).get_mean()
        //           << std::endl;
        // std::cout << "lambda " << lambda << std::endl;
        lambdas.push_back(lambda);
    }

    /*
    for (uint32_t i = 0; i < dictionary_size; i++) {
        std::cout << "Working on " << i << std::endl;
        print_buckets(*time_bin_result.sample_per_index.find(i),
    num_buckets);
    }*/

    while (result_options.size() < 32) {
	    result_options.push_back(database.get_code_dictionary().size());
	    lambdas.push_back(1e-6);
    }

    json j;
    j["codes"] = result_options;
    j["time_bins"] = get_buckets(global_samples, num_buckets);
    j["lambdas"] = lambdas;

    std::vector<std::uint8_t> v = json::to_msgpack(j);

    std::ofstream o(output, std::ios_base::binary);

    o.write((const char*)v.data(), v.size());
}
