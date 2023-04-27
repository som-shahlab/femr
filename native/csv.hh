#pragma once

#include <boost/filesystem.hpp>
#include <deque>
#include <fstream>
#include <iostream>
#include <vector>

#include "absl/strings/substitute.h"
#include "zstd.h"

class ZSTDCFree {
   public:
    void operator()(ZSTD_CStream* ptr);
};

class ZstdWriter {
   public:
    explicit ZstdWriter(const boost::filesystem::path& filename);

    void add_data(std::string_view data);

    ~ZstdWriter();

    boost::filesystem::path fname;

   private:
    void flush(bool final = false);

    std::ofstream f;
    std::unique_ptr<ZSTD_CStream, ZSTDCFree> stream;
    std::vector<char> in_buffer_data;
    std::vector<char> out_buffer_data;

    size_t in_buffer_pos;
};

class TextWriter {
   public:
    explicit TextWriter(const boost::filesystem::path& filename);

    void add_data(std::string_view data);

    boost::filesystem::path fname;

   private:
    std::ofstream f;
};

class ZSTDDFree {
   public:
    void operator()(ZSTD_DStream* ptr);
};

class TextReader {
   public:
    explicit TextReader(const boost::filesystem::path& filename);

    std::string_view get_data() const;

    void seek(size_t seek_amount);

    bool eof() const;
    boost::filesystem::path fname;

   private:
    std::ifstream f;
    std::vector<char> buffer_data;
    size_t buffer_end;
};

class ZstdReader {
   public:
    explicit ZstdReader(const boost::filesystem::path& filename);

    std::string_view get_data() const;

    void seek(size_t seek_amount);

    bool eof() const;
    boost::filesystem::path fname;

   private:
    TextReader reader;
    std::unique_ptr<ZSTD_DStream, ZSTDDFree> stream;
    std::vector<char> out_buffer_data;
    size_t out_buffer_end;

    std::string_view current_in;
};

template <typename Writer>
class CSVWriter {
   public:
    CSVWriter(const boost::filesystem::path& filename,
              const std::vector<std::string>& columns, char _delimiter)
        : writer(filename),
          num_columns(columns.size()),
          delimiter_str(std::string_view(&_delimiter, 1)) {
        add_row(columns);
    }

    void add_row(const std::vector<std::string>& columns) {
        if (columns.size() != num_columns) {
            throw std::runtime_error("Wrong number of columns? " +
                                     std::to_string(columns.size()));
        }
        bool first = true;
        for (const auto& column : columns) {
            if (!first) {
                writer.add_data(delimiter_str);
            } else {
                first = false;
            }

            bool found_any = false;
            temp.clear();
            temp.push_back('"');

            if (!column.empty() &&
                (column.front() == ' ' || column.back() == ' ')) {
                found_any = true;
            }

            for (const auto& c : column) {
                if (c != '"') {
                    found_any = found_any || c == ',' || c == '\n';
                    temp.push_back(c);
                } else {
                    found_any = true;
                    temp.push_back('"');
                    temp.push_back('"');
                }
            }

            temp.push_back('"');

            if (found_any) {
                writer.add_data(temp);
            } else {
                writer.add_data(
                    std::string_view(temp.data() + 1, temp.size() - 2));
            }
        }
        writer.add_data("\n");
    }

   private:
    Writer writer;
    size_t num_columns;
    std::string delimiter_str;
    std::string temp;
};

template <typename F>
inline size_t line_iter(std::string_view line, char delimiter, F f) {
    size_t line_index = 0;
    bool in_quotes = false;
    size_t index = 0;
    std::string* field_str = f(index++);

    while (line_index < line.size()) {
        char next_char = line[line_index++];

        char future = '\0';
        if (line_index < line.size()) {
            future = line[line_index];
        }

        if (in_quotes) {
            if (next_char == '"') {
                if (future == '"') {
                    line_index++;
                    if (field_str != nullptr) {
                        field_str->push_back('"');
                    }
                } else {
                    in_quotes = false;
                }
            } else {
                if (field_str != nullptr) {
                    field_str->push_back(next_char);
                }
            }
        } else {
            if (next_char == '\r') {
                if (future == '\n') {
                    return line_index + 1;
                } else {
                    return 0;
                }
            } else if (next_char == '\n') {
                return line_index;
            } else if (next_char == delimiter) {
                field_str = f(index++);
            } else if (next_char == '"') {
                in_quotes = true;
            } else {
                if (field_str != nullptr) {
                    field_str->push_back(next_char);
                }
            }
        }
    }

    return 0;
}

template <typename Reader>
std::vector<std::string> get_csv_columns_from_reader(Reader& reader,
                                                     char delimiter) {
    std::string_view line = reader.get_data();
    std::vector<std::string> result;

    size_t increment = line_iter(line, delimiter, [&result](size_t index) {
        result.emplace_back("");
        return &result[index];
    });
    if (increment == 0) {
        throw std::runtime_error("Could not even load the header? " +
                                 reader.fname.string());
    }
    reader.seek(increment);

    return result;
}

inline std::vector<std::string> get_csv_columns(
    const boost::filesystem::path& filename, char delimiter) {
    if (filename.extension() == ".zst") {
        ZstdReader reader(filename);
        return get_csv_columns_from_reader(reader, delimiter);
    } else {
        TextReader reader(filename);
        return get_csv_columns_from_reader(reader, delimiter);
    }
}

template <typename Reader>
class CSVReader {
   public:
    CSVReader(const boost::filesystem::path& filename, char _delimiter)
        : reader(filename), delimiter(_delimiter) {
        std::vector<std::string> file_columns =
            get_csv_columns_from_reader(reader, delimiter);
        columns = file_columns;
        init_helper(file_columns);
    }

    CSVReader(const boost::filesystem::path& filename,
              const std::vector<std::string>& _columns, char _delimiter)
        : columns(_columns), reader(filename), delimiter(_delimiter) {
        std::vector<std::string> file_columns =
            get_csv_columns_from_reader(reader, delimiter);
        init_helper(file_columns);
    }

    std::vector<std::string>& get_row() {
        for (const auto& item : current_row_set) {
            if (!item) {
                throw std::runtime_error(
                    "Some items not set but about to read?");
            }
        }
        return current_row;
    }

    bool next_row() {
        std::string_view line = reader.get_data().substr(current_offset);

        current_row.resize(current_row_set.size());

        for (auto& column : current_row) {
            column.clear();
        }

        for (auto& item : current_row_set) {
            item = false;
        }

        size_t increment =
            line_iter(line, delimiter, [this](size_t index) -> std::string* {
                ssize_t actual_index = column_map.at(index);
                if (actual_index < 0) {
                    return nullptr;
                } else {
                    current_row_set[actual_index] = true;
                    return &current_row[actual_index];
                }
            });

        if (increment == 0) {
            if (reader.eof()) {
                return false;
            } else {
                reader.seek(current_offset);
                current_offset = 0;
                return next_row();
            }
        } else {
            for (const auto& item : current_row_set) {
                if (!item) {
                    throw std::runtime_error(
                        "Some items not set? " +
                        std::string(line.substr(0, increment)));
                }
            }

            current_offset += increment;
            return true;
        }
    }

    std::vector<std::string> columns;

   private:
    void init_helper(const std::vector<std::string>& file_columns) {
        current_row.resize(columns.size());
        current_row_set.resize(columns.size());
        column_map.reserve(file_columns.size());

        std::vector<bool> found_column(columns.size(), false);

        for (const auto& file_column : file_columns) {
            bool found = false;
            for (size_t i = 0; i < columns.size(); i++) {
                if (columns[i] == file_column) {
                    if (found || found_column[i]) {
                        throw std::runtime_error(
                            absl::StrCat("Duplicate column? ", columns[i]));
                    }
                    found = true;
                    found_column[i] = true;
                    column_map.push_back(i);
                }
            }
            if (!found) {
                column_map.push_back(-1);
            }
        }

        for (size_t i = 0; i < columns.size(); i++) {
            if (!found_column[i]) {
                throw std::runtime_error(absl::StrCat(
                    "Unable to find column '", columns[i], "' in '",
                    absl::StrJoin(file_columns, "','"), "'"));
            }
        }

        current_offset = 0;
    }

    size_t current_offset;
    Reader reader;
    std::vector<ssize_t> column_map;
    std::vector<std::string> current_row;
    std::deque<bool> current_row_set;
    char delimiter;
};
