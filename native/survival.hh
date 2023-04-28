#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "database.hh"
#include <iostream>
#include "flatmap.hh"

struct SurvivalEvent {
    std::vector<uint32_t> times;
    uint32_t code;
};

inline bool should_make_prediction(uint32_t last_prediction_age,
                                   uint32_t current_age,
                                   boost::optional<uint32_t> next_age,
                                   int current_year) {
    if (current_year < 2010) {
        // Only make predictions when we have OK data
        return false;
    }

    if (!next_age) {
        return false;
    }
    if (*next_age < 2 * 60 * 24) {
        // Don't try to predict stuff on the day of birth
        return false;
    }

    if (current_age == *next_age) {
        // Dont make duplicate predictions
        return false;
    }
    if (current_age - last_prediction_age < (60 * 24)) {
        return false;
    }

    return true;
}

struct SurvivalCalculator {
    FlatMap<std::vector<uint32_t>> future_times;
    std::vector<SurvivalEvent> survival_events;
    uint32_t final_age;

    void preprocess_patient(
        const Patient& p,
        const FlatMap<std::vector<uint32_t>>& survival_dictionary,
        uint32_t max_age = std::numeric_limits<uint32_t>::max()) {
        preprocess_patient(
            p,
            [&](uint32_t code) {
                auto iter = survival_dictionary.find(code);
                if (iter == nullptr) {
                    return absl::Span<const uint32_t>();
                } else {
                    return absl::Span<const uint32_t>(*iter);
                }
            },
            max_age);
    }

    template <typename F>
    void preprocess_patient(
        const Patient& p, F get_all_parents,
        uint32_t max_age = std::numeric_limits<uint32_t>::max()) {
        future_times.clear();
        survival_events.clear();

        for (const Event& event : p.events) {
            if (event.value_type == ValueType::NONE) {
                for (const auto& parent : get_all_parents(event.code)) {
                    std::vector<uint32_t>* item = future_times.find_or_insert(
                        parent, std::vector<uint32_t>());
                    item->push_back(event.start_age_in_minutes);
                }
            }
        }

        final_age = p.events[p.events.size() - 1].start_age_in_minutes;
        final_age = std::min(final_age, max_age);

        for (uint32_t code : future_times.keys()) {
            auto* item = future_times.find(code);

            SurvivalEvent event;
            event.code = code;
            event.times = std::move(*item);
            assert(event.times.size() > 0);

            event.times.erase(
                std::remove_if(
                    event.times.begin(), event.times.end(),
                    [this](uint32_t time) { return time > final_age; }),
                event.times.end());

            std::sort(std::begin(event.times), std::end(event.times),
                      std::greater<uint32_t>());

            if (event.times.size() > 0) {
                survival_events.emplace_back(std::move(event));
            }
        }
    }

    std::pair<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>>
    get_times_for_event(uint32_t event_age) {
        std::vector<std::pair<uint32_t, uint32_t>> events;

        size_t current_index = 0;
        for (size_t i = 0; i < survival_events.size(); i++) {
            SurvivalEvent entry = std::move(survival_events[i]);
            while (entry.times.size() > 0 && entry.times.back() <= event_age) {
                entry.times.pop_back();
            }

            if (entry.times.size() > 0) {
                events.push_back(
                    std::make_pair(entry.times.back() - event_age, entry.code));
                survival_events[current_index++] = std::move(entry);
            }
        }

        survival_events.resize(current_index);
        return std::make_pair(final_age - event_age, std::move(events));
    }
};

inline FlatMap<std::vector<uint32_t>> get_mapping(
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
