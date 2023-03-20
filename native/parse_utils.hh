#ifndef PARSE_UTILS_H_INCLUDED
#define PARSE_UTILS_H_INCLUDED

#include <fstream>
#include <iostream>
#include "absl/strings/numbers.h"
#include "absl/strings/substitute.h"
#include "absl/time/civil_time.h"

template <typename T>
void attempt_parse_or_die(std::string_view text, T& value) {
    if (!absl::SimpleAtoi(text, &value)) {
        std::cout << "Not time\n";
        std::cout << std::string(text) << '\n';
        throw std::runtime_error("A) Could not parse " + std::string(text));
    }
}

template <typename T>
void attempt_parse_time_or_die(std::string_view text, T& value) {
    if (!absl::ParseCivilTime(text, &value)) {
        std::cout << "Time\n";
        std::cout << std::string(text) << '\n';
        throw std::runtime_error("B) Could not parse time " + std::string(text));
    }
}

#endif