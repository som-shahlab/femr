#ifndef PARSE_UTILS_H_INCLUDED
#define PARSE_UTILS_H_INCLUDED

#include <iostream>

#include "absl/strings/numbers.h"
#include "absl/strings/substitute.h"

template <typename T>
void attempt_parse_or_die(std::string_view text, T& value) {
    if (!absl::SimpleAtoi(text, &value)) {
        std::cout << absl::Substitute("Could not parse $0\n", text);
        abort();
    }
}

#endif