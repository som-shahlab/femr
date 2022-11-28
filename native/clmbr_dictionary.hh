#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>

// The type of dictionary entry
enum class DictEntryType {
    CODE,
    NUMERIC,
    TEXT,
};

// The exact dictionary entry
struct DictEntry {
    DictEntryType type;
    uint32_t code;
    double weight;

    // Only for numeric dictionary entries
    // A numeric dictionary entry covers from [start_val, end_val)
    double val_start = 0;
    double val_end = 0;

    // Only for text dictionary entries
    uint32_t text_value = 0;

    bool operator<(const DictEntry& other) { return weight < other.weight; }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DictEntry, type, code, weight, val_start,
                                   val_end, text_value)
};
