#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "database.hh"
#include "flatmap.hh"

using json = nlohmann::json;

bool limit = false;

template <typename T>
std::vector<uint32_t> convert_to_ranks(const T& t) {
    T vals = t;
    std::sort(std::begin(vals), std::end(vals));

    std::vector<uint32_t> result;
    result.reserve(t.size());

    for (const auto& entry : t) {
        uint32_t rank =
            (std::lower_bound(std::begin(vals), std::end(vals), entry) -
             std::begin(vals));
        result.emplace_back(rank);
    }

    return result;
}

template <typename T>
double mean(const T& a) {
    double result = 0;
    for (const auto& e : a) {
        result += e;
    }
    return result / a.size();
}

template <typename T>
double compute_spearman(const T& a, const T& b) {
    if (a.size() == 0 && b.size() == 0) {
        return 1;
    }
    auto a_ranks = convert_to_ranks(a);
    auto b_ranks = convert_to_ranks(b);

    a_ranks.push_back(1);
    b_ranks.push_back(1);

    double mean_a = mean(a_ranks);
    double mean_b = mean(b_ranks);

    double numerator = 0;
    double first = 0;
    double second = 0;

    // std::cout << "Means " << mean_a << " " << mean_b << std::endl;

    for (size_t i = 0; i < a.size(); i++) {
        double a_delta = (a_ranks[i] - mean_a);
        double b_delta = (b_ranks[i] - mean_b);
        numerator += a_delta * b_delta;
        first += a_delta * a_delta;
        second += b_delta * b_delta;
    }
    //    std::cout << "Temp " << first << " " << second << " " << numerator
    //            << std::endl;

    return numerator / std::sqrt(first * second);
}

int main() {
    boost::filesystem::path path =
        "/share/pi/nigam/data/"
        "som-rit-phi-starr-prod.starr_omop_cdm5_deid_1pcent_2022_09_05_extract";
    // boost::filesystem::path path =
    // "/local-scratch/nigam/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05";
    PatientDatabase data(path, true);

    auto& dict = data.get_code_dictionary();

    absl::flat_hash_map<uint32_t, absl::flat_hash_map<uint32_t, uint32_t>>
        counts;

    PatientDatabaseIterator iterator = data.iterator();

    uint32_t starting_count = 0;

    for (size_t i = 0; i < data.size(); i++) {
        const Patient& patient = iterator.get_patient(i);

        std::vector<uint32_t> current_codes;

        if (limit && i > 100) {
            break;
        }

        auto flush = [&]() {
            for (const auto& code : current_codes) {
                starting_count += 1;
                for (const auto& code_b : current_codes) {
                    counts[code][code_b] += 1;
                }
            }
        };

        float last_age = -1;
        for (size_t event_index = 0; event_index < patient.events.size();
             event_index++) {
            const Event& event = patient.events[event_index];
            float age = (float)event.age_in_days +
                        (float)event.minutes_offset / (24 * 60.0);

            if (age != last_age && last_age != -1) {
                flush();
                current_codes.clear();
            }

            current_codes.push_back(event.code);

            last_age = age;
        }

        flush();
    }

    struct MatchTemp {
        uint32_t total_matches = 0;
        uint32_t invalid_matches = 0;
        std::vector<float> numeric_matches_a = {};
        std::vector<float> numeric_matches_b = {};
        std::vector<uint32_t> text_matches_a = {};
        std::vector<uint32_t> text_matches_b = {};
    };

    absl::flat_hash_set<uint32_t> removed;
    absl::flat_hash_map<std::pair<uint32_t, uint32_t>, MatchTemp> match_info;

    for (const auto& entry : counts) {
        uint32_t self_count = entry.second.find(entry.first)->second;
        if (self_count <= 1000) {
            // Not enough samples;
            continue;
        }
        for (const auto& other_entry : entry.second) {
            if (entry.first == other_entry.first) {
                continue;
            }

            float fraction = (float)other_entry.second / (float)self_count;

            float other_fraction = (float)other_entry.second /
                                   (float)counts.find(other_entry.first)
                                       ->second.find(other_entry.first)
                                       ->second;

            if (fraction > 0.99 && other_fraction > 0.99) {
                removed.insert(entry.first);
                removed.insert(other_entry.first);
                MatchTemp temp;
                match_info.insert(std::make_pair(
                    std::make_pair(entry.first, other_entry.first), temp));

                match_info.insert(std::make_pair(
                    std::make_pair(other_entry.first, entry.first), temp));

                break;
            }
        }
    }

    for (size_t i = 0; i < data.size(); i++) {
        const Patient& patient = iterator.get_patient(i);

        std::vector<Event> current_events;

        if (limit && i > 100) {
            break;
        }

        auto flush = [&]() {
            for (const auto& code : current_events) {
                for (const auto& code_b : current_events) {
                    auto iter =
                        match_info.find(std::make_pair(code.code, code_b.code));
                    if (iter == std::end(match_info)) {
                        continue;
                    }
                    auto& info = iter->second;

                    info.total_matches += 1;
                    if (code.value_type != code_b.value_type) {
                        info.invalid_matches += 1;
                    } else {
                        switch (code.value_type) {
                            case ValueType::NONE:
                                break;
                            case ValueType::UNIQUE_TEXT:
                                info.invalid_matches += 1;
                                break;
                            case ValueType::SHARED_TEXT:
                                info.text_matches_a.emplace_back(
                                    code.text_value);
                                info.text_matches_b.emplace_back(
                                    code_b.text_value);
                                break;
                            case ValueType::NUMERIC:
                                info.numeric_matches_a.emplace_back(
                                    code.numeric_value);
                                info.numeric_matches_b.emplace_back(
                                    code_b.numeric_value);
                                break;
                            default:
                                throw std::runtime_error("Case not handled");
                        }
                    }
                }
            }
        };

        float last_age = -1;
        for (size_t event_index = 0; event_index < patient.events.size();
             event_index++) {
            const Event& event = patient.events[event_index];
            float age = (float)event.age_in_days +
                        (float)event.minutes_offset / (24 * 60.0);

            if (age != last_age && last_age != -1) {
                flush();
                current_events.clear();
            }

            if (removed.count(event.code)) {
                current_events.push_back(event);
            }

            last_age = age;
        }

        flush();
    }

    absl::flat_hash_set<uint32_t> actually_removed;
    std::vector<std::string> result;

    size_t total_removed = 0;

    for (const auto& entry : match_info) {
        const auto& codes = entry.first;
        if (actually_removed.count(codes.first) ||
            actually_removed.count(codes.second)) {
            continue;
        }

        const auto& info = entry.second;

        double numeric_spear =
            compute_spearman(info.numeric_matches_a, info.numeric_matches_b);

        double text_spear =
            compute_spearman(info.text_matches_a, info.text_matches_b);

        double frac_invalid =
            (float)info.invalid_matches / (float)info.total_matches;
        double frac_numeric =
            info.numeric_matches_a.size() / (float)info.total_matches;
        double frac_text =
            info.text_matches_a.size() / (float)info.total_matches;
        double frac_none = 1 - frac_invalid - frac_numeric - frac_text;

        double total_spear = 1 * frac_none + 0 * frac_invalid +
                             numeric_spear * frac_numeric +
                             text_spear * frac_text;

        if (total_spear < 0.98) {
            continue;
        }

        uint32_t num_removed =
            counts.find(codes.first)->second.find(codes.first)->second;
        std::cout << dict[codes.first] << " " << dict[codes.second] << " "
                  << num_removed << " (" << frac_none << ", " << frac_text
                  << "[" << text_spear << "], " << frac_numeric << "["
                  << numeric_spear << "])" << std::endl;
        actually_removed.insert(codes.first);
	result.push_back(std::string(dict[codes.first]));
        total_removed += num_removed;
    }

    std::cout << total_removed << " " << starting_count << std::endl;

    std::cout << json(result) <<std::endl;
}
