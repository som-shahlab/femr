#include "parse_utils.h"

absl::CivilDay parse_date(std::string_view datestr) {
    std::string_view time_column = datestr;
    auto location = time_column.find(' ');
    if (location != std::string_view::npos) {
        time_column = time_column.substr(0, location);
    }

    location = time_column.find('T');
    if (location != std::string_view::npos) {
        time_column = time_column.substr(0, location);
    }

    auto first_dash = time_column.find('-');
    int year;
    attempt_parse_or_die(time_column.substr(0, first_dash), year);
    time_column = time_column.substr(first_dash + 1, std::string_view::npos);

    auto second_dash = time_column.find('-');
    int month;
    attempt_parse_or_die(time_column.substr(0, second_dash), month);
    time_column = time_column.substr(second_dash + 1, std::string_view::npos);

    int day;
    attempt_parse_or_die(time_column, day);

    return absl::CivilDay(year, month, day);
}