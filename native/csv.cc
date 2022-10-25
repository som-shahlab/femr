#include "csv.hh"

#include <fstream>

#include "absl/strings/substitute.h"
#include "zstd.h"

namespace {
const int BUFFER_SIZE = 1024 * 1024;

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

}  // namespace

void ZSTDCFree::operator()(ZSTD_CStream* ptr) { ZSTD_freeCStream(ptr); }

ZstdWriter::ZstdWriter(const boost::filesystem::path& filename) {
    f.rdbuf()->pubsetbuf(nullptr, 0);
    f.open(filename.c_str());

    stream.reset(ZSTD_createCStream());
    ZSTD_initCStream(stream.get(), 1);

    in_buffer_data.resize(BUFFER_SIZE * 2);
    out_buffer_data.resize(ZSTD_compressBound(BUFFER_SIZE));

    in_buffer_pos = 0;
}

void ZstdWriter::add_data(std::string_view data) {
    if (data.size() > BUFFER_SIZE) {
        throw std::runtime_error(
            "Cannot process data greater than BUFFER_SIZE");
    }
    if (in_buffer_pos + data.size() > (BUFFER_SIZE * 2)) {
        throw std::runtime_error("Should never happen, buffsize failure");
    }
    std::memcpy(in_buffer_data.data() + in_buffer_pos, data.data(),
                data.size());
    in_buffer_pos += data.size();
    if (in_buffer_pos >= BUFFER_SIZE) {
        flush();
    }
}

void ZstdWriter::flush(bool final) {
    ZSTD_EndDirective op;
    if (final) {
        op = ZSTD_e_end;
    } else {
        op = ZSTD_e_continue;
    }

    ZSTD_inBuffer in_buffer = {in_buffer_data.data(), in_buffer_pos, 0};
    ZSTD_outBuffer out_buffer = {out_buffer_data.data(), BUFFER_SIZE, 0};

    int ret = ZSTD_compressStream2(stream.get(), &out_buffer, &in_buffer, op);

    if (ret != 0) {
        throw std::runtime_error("A single one should always be good enough");
    }

    f.write(out_buffer_data.data(), out_buffer.pos);
    std::memmove(in_buffer_data.data(), in_buffer_data.data() + in_buffer.pos,
                 in_buffer_pos - in_buffer.pos);

    in_buffer_pos -= in_buffer.pos;
}

ZstdWriter::~ZstdWriter() { flush(true); }

void ZSTDDFree::operator()(ZSTD_DStream* ptr) { ZSTD_freeDStream(ptr); }

ZstdReader::ZstdReader(const boost::filesystem::path& filename) {
    f.rdbuf()->pubsetbuf(nullptr, 0);
    f.open(filename.c_str());

    stream.reset(ZSTD_createDStream());
    ZSTD_initDStream(stream.get());

    in_buffer_data.resize(BUFFER_SIZE * 2);
    out_buffer_data.resize(BUFFER_SIZE);

    out_buffer_end = 0;

    f.read(in_buffer_data.data(), BUFFER_SIZE * 2);
    in_buffer_start = 0;
    in_buffer_end = f.gcount();
    seek(0);
}

std::string_view ZstdReader::get_data() {
    return std::string_view(out_buffer_data.data(), out_buffer_end);
}

void ZstdReader::seek(size_t seek_amount) {
    std::memmove(out_buffer_data.data(), out_buffer_data.data() + seek_amount,
                 out_buffer_end - seek_amount);

    out_buffer_end -= seek_amount;

    if (((in_buffer_end - in_buffer_start) < BUFFER_SIZE) && !f.eof()) {
        std::memmove(in_buffer_data.data(),
                     in_buffer_data.data() + in_buffer_start,
                     in_buffer_end - in_buffer_start);
        in_buffer_end -= in_buffer_start;
        in_buffer_start = 0;

        f.read(in_buffer_data.data() + in_buffer_end,
               BUFFER_SIZE * 2 - in_buffer_end);
        in_buffer_end += f.gcount();
    }

    ZSTD_inBuffer in_buffer = {.src = in_buffer_data.data(),
                               .size = in_buffer_end,
                               .pos = in_buffer_start};
    ZSTD_outBuffer out_buffer = {.dst = out_buffer_data.data(),
                                 .size = BUFFER_SIZE,
                                 .pos = out_buffer_end};
    size_t ret = ZSTD_decompressStream(stream.get(), &out_buffer, &in_buffer);
    if (ZSTD_isError(ret) != 0) {
        throw std::runtime_error("Got error while decompressing? " +
                                 std::string(ZSTD_getErrorName(ret)));
    }

    in_buffer_start = in_buffer.pos;
    out_buffer_end = out_buffer.pos;
}

bool ZstdReader::eof() const {
    return (in_buffer_end == in_buffer_start) && f.eof();
}

CSVWriter::CSVWriter(const boost::filesystem::path& filename,
                     const std::vector<std::string>& columns, char _delimiter)
    : writer(filename),
      num_columns(columns.size()),
      delimiter_str(std::string_view(&_delimiter, 1)) {
    add_row(columns);
}

void CSVWriter::add_row(const std::vector<std::string>& columns) {
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
            writer.add_data(std::string_view(temp.data() + 1, temp.size() - 2));
        }
    }
    writer.add_data("\n");
}

std::vector<std::string> get_csv_columns(
    const boost::filesystem::path& filename, char delimiter) {
    ZstdReader reader(filename);
    return get_csv_columns(reader, delimiter);
}

std::vector<std::string> get_csv_columns(ZstdReader& reader, char delimiter) {
    std::string_view line = reader.get_data();
    std::vector<std::string> result;

    size_t increment = line_iter(line, delimiter, [&result](size_t index) {
        result.emplace_back("");
        return &result[index];
    });
    if (increment == 0) {
        throw std::runtime_error("Could not even load the header?");
    }
    reader.seek(increment);

    return result;
}

CSVReader::CSVReader(const boost::filesystem::path& filename,
                     const std::vector<std::string>& columns, char _delimiter)
    : reader(filename), delimiter(_delimiter) {
    std::vector<std::string> file_columns = get_csv_columns(reader, delimiter);
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
            throw std::runtime_error(
                absl::StrCat("Unable to find column '", columns[i], "' in '",
                             absl::StrJoin(file_columns, "','"), "'"));
        }
    }

    current_offset = 0;
}

std::vector<std::string>& CSVReader::get_row() {
    for (const auto& item : current_row_set) {
        if (!item) {
            throw std::runtime_error("Some items not set but about to read?");
        }
    }
    return current_row;
}

bool CSVReader::next_row() {
    std::string_view line = reader.get_data().substr(current_offset);

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
