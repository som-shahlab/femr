#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "database.hh"
#include "flatmap.hh"

struct SurvivalEvent {
    std::vector<double> times;
    uint32_t code;
};

inline bool should_make_prediction(double last_prediction_age,
                                   double current_age, double next_age,
                                   int current_year) {
    if (current_year < 2010) {
        // Only make predictions when we have OK data
        return false;
    }
    if (next_age < 2 || std::isinf(next_age)) {
        // Don't try to predict stuff on the day of birth
        return false;
    }
    if (current_age == next_age) {
        // Dont make duplicate predictions
        return false;
    }
    if (current_age - last_prediction_age < 1) {
        return false;
    }

    return true;
}

struct SurvivalCalculator {
    FlatMap<std::vector<double>> future_times;
    std::vector<SurvivalEvent> survival_events;
    double final_age;

    void preprocess_patient(
        const Patient& p,
        const FlatMap<std::vector<uint32_t>>& survival_dictionary) {
        preprocess_patient(p, [&](uint32_t code) {
            auto iter = survival_dictionary.find(code);
            if (iter == nullptr) {
                return absl::Span<const uint32_t>();
            } else {
                return absl::Span<const uint32_t>(*iter);
            }
        });
    }

    template <typename F>
    void preprocess_patient(const Patient& p, F get_all_parents) {
        future_times.clear();
        survival_events.clear();

        for (const Event& event : p.events) {
            if (event.value_type == ValueType::NONE) {
                for (const auto& parent : get_all_parents(event.code)) {
                    std::vector<double>* item = future_times.find_or_insert(
                        parent, std::vector<double>());
                    item->push_back(event.age);
                }
            }
        }
        final_age = p.events[p.events.size() - 1].age;

        for (uint32_t code : future_times.keys()) {
            auto* item = future_times.find(code);

            SurvivalEvent event;
            event.code = code;
            event.times = std::move(*item);
            assert(event.times.size() > 0);

            std::sort(std::begin(event.times), std::end(event.times),
                      std::greater<double>());

            for (const auto& t : event.times) {
                if (t > 365 * 120) {
                    std::cout << "Should never happen" << std::endl;
                    abort();
                }
            }

            survival_events.emplace_back(std::move(event));
        }
    }

    std::pair<double, std::vector<std::pair<double, uint32_t>>>
    get_times_for_event(double event_age) {
        std::vector<std::pair<double, uint32_t>> events;

        size_t current_index = 0;
        for (size_t i = 0; i < survival_events.size(); i++) {
            SurvivalEvent entry = std::move(survival_events[i]);
            while (entry.times.size() > 0 && entry.times.back() <= event_age) {
                entry.times.pop_back();
            }

            for (const auto& t : entry.times) {
                if (t > 365 * 120) {
                    std::cout << "Should never happen second" << std::endl;
                    abort();
                }
            }

            if (entry.times.size() > 0) {
                events.push_back(
                    std::make_pair(entry.times.back() - event_age, entry.code));
                survival_events[current_index++] = std::move(entry);
            }
        }

        for (const auto& t : events) {
            if (t.first > 365 * 120) {
                std::cout << "Should never happen second" << std::endl;
                abort();
            }
        }

        survival_events.resize(current_index);
        return std::make_pair(final_age - event_age, std::move(events));
    }
};

FlatMap<std::vector<uint32_t>> get_mapping(
    Ontology& ontology, const std::vector<uint32_t>& survival_codes) {
    FlatMap<std::vector<uint32_t>> survival_dictionary;
    FlatMap<uint32_t> to_index;
    for (uint32_t i = 0; i < survival_codes.size(); i++) {
        uint32_t code = survival_codes[i];
        to_index.insert(code, i);
    }

    for (uint32_t code = 0; code < ontology.get_dictionary().size(); code++) {
        std::vector<uint32_t> result;
        for (uint32_t parent : ontology.get_all_parents(code)) {
            auto iter = to_index.find(parent);
            if (iter == nullptr) {
                continue;
            }
            result.push_back(*iter);
        }

        if (result.size() > 0) {
            survival_dictionary.insert(code, result);
        }
    }
    return survival_dictionary;
}
