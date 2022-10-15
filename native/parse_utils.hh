#ifndef PARSE_UTILS_H_INCLUDED
#define PARSE_UTILS_H_INCLUDED

#include "absl/strings/numbers.h"
#include "absl/strings/substitute.h"
#include "absl/time/civil_time.h"

template <typename T>
void attempt_parse_or_die(std::string_view text, T& value) {
    if (!absl::SimpleAtoi(text, &value)) {
        throw std::runtime_error("Could not parse " + std::string(text));
    }
}

absl::CivilDay parse_date(std::string_view datestr);

#endif
