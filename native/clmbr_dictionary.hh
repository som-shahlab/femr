#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>

// The type of dictionary entry
enum class DictEntryType {
    CODE = 0,
    NUMERIC = 1,
    TEXT = 2,
    UNUSED = 3,
};

// The exact dictionary entry
struct DictEntry {
    DictEntryType type;
    std::string code_string;
    double weight;

    // Only for numeric dictionary entries
    // A numeric dictionary entry covers from [start_val, end_val)
    double val_start = 0;
    double val_end = 0;

    // Only for text dictionary entries
    std::string text_string;

    bool operator<(const DictEntry& other) const { return weight < other.weight; }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DictEntry, type, code_string, weight, val_start,
                                   val_end, text_string)
};
