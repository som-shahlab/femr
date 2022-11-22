#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>

enum class DictEntryType {
    CODE,
    NUMERIC,
    TEXT,
};

struct DictEntry {
    DictEntryType type;
    uint32_t code;
    double weight;

    double val_start = 0;
    double val_end = 0;
    uint32_t text_value = 0;

    bool operator<(const DictEntry& other) { return weight < other.weight; }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(DictEntry, type, code, weight, val_start,
                                   val_end, text_value)
};
