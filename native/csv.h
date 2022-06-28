#ifndef CSV_H_INCLUDED
#define CSV_H_INCLUDED

#define WITH_GZFILEOP
#include "absl/strings/substitute.h"
#include "zlib.h"

const int BUFFER_SIZE = 1024 * 1024 * 10;

template <typename F>
inline void line_iter(const char* line, char delimiter, bool quotes, F f) {
    const char* iter = line;
    int index = 0;
    const char* field_start = iter;
    int field_length = 0;

    bool in_quotes = false;

    while (true) {
        char next_char = *iter++;

        if (in_quotes) {
            field_length++;
            if (next_char == '"') {
                in_quotes = false;
            }
        } else {
            if (next_char == '\n' || next_char == '\r') {
                f(index++, std::string_view(field_start, field_length));
                break;
            } else if (next_char == '\0') {
                std::cout << absl::Substitute("Line without newline? $0", line)
                          << std::endl;
                abort();
            } else if (next_char == delimiter) {
                f(index++, std::string_view(field_start, field_length));
                field_length = 0;
                field_start = iter;
            } else if (next_char == '"' && quotes) {
                in_quotes = true;
                field_length++;
            } else {
                field_length++;
            }
        }
    }
}

template <typename F>
inline void line_iter(const char* line, char delimiter, F f) {
    line_iter(line, delimiter, true, f);
}

template <typename F>
inline void csv_iterator(const char* filename,
                         std::vector<std::string_view> columns, char delimiter,
                         std::optional<int> limit, bool quotes,
                         bool case_sensitive, F f) {
    gzFile file = gzopen(filename, "r");
    if (file == nullptr) {
        std::cout << absl::Substitute("Could not open $0 due to $1", filename,
                                      std::strerror(errno))
                  << std::endl;
        ;
        abort();
    }

    gzbuffer(file, BUFFER_SIZE);

    std::vector<char> buffer(BUFFER_SIZE);
    char* first_line = gzgets(file, buffer.data(), BUFFER_SIZE);

    if (first_line == nullptr) {
        std::cout << absl::Substitute("Could read header on $0 due to $1\n",
                                      file, std::strerror(errno));
        abort();
    }

    std::vector<std::string_view> all_columns;
    line_iter(first_line, delimiter, quotes,
              [&](int index, std::string_view column) {
                  all_columns.push_back(column);
              });

    std::vector<int> index_map(all_columns.size(), -1);

    for (size_t i = 0; i < columns.size(); i++) {
        bool found = false;
        for (size_t index = 0; index < all_columns.size(); index++) {
            bool is_same;
            if (case_sensitive) {
                is_same = all_columns[index] == columns[i];
            } else {
                is_same =
                    absl::EqualsIgnoreCase(all_columns[index], columns[i]);
            }

            if (is_same) {
                index_map[index] = i;
                found = true;
            }
        }

        if (!found) {
            std::cout
                << absl::Substitute(
                       "Could not find column $0 in file $1 with delimiter $2",
                       columns[i], filename, delimiter)
                << std::endl;
            ;
            std::cout << absl::Substitute("Had columns ($0): ", all_columns.size());
            for (auto col : all_columns) {
                std::cout << absl::Substitute("\"$0\",", col);
            }
            std::cout << std::endl << std::endl << first_line << std::endl;
            abort();
        }
    }

    std::vector<std::string_view> output_columns(columns.size());

    while (true) {
        char* next_line = gzgets(file, buffer.data(), BUFFER_SIZE);
        if (next_line == nullptr) {
            break;
        }

        line_iter(
            next_line, delimiter, quotes,
            [&index_map, &output_columns](int index, std::string_view column) {
                int desired_index = index_map[index];
                if (desired_index != -1) {
                    output_columns[desired_index] = column;
                }
            });

        f(output_columns);
    }

    gzclose(file);
}

template <typename F>
inline void csv_iterator(const char* filename,
                         std::vector<std::string_view> columns, char delimiter,
                         std::optional<int> limit, F f) {
    csv_iterator(filename, columns, delimiter, limit, true, true, f);
}

#endif