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

   private:
    void flush(bool final = false);

    std::ofstream f;
    std::unique_ptr<ZSTD_CStream, ZSTDCFree> stream;
    std::vector<char> in_buffer_data;
    std::vector<char> out_buffer_data;

    size_t in_buffer_pos;
};

class ZSTDDFree {
   public:
    void operator()(ZSTD_DStream* ptr);
};

class ZstdReader {
   public:
    explicit ZstdReader(const boost::filesystem::path& filename);

    std::string_view get_data();

    void seek(size_t seek_amount);

    bool eof() const;

   private:
    std::ifstream f;
    std::unique_ptr<ZSTD_DStream, ZSTDDFree> stream;
    std::vector<char> in_buffer_data;
    std::vector<char> out_buffer_data;

    size_t in_buffer_start;
    size_t in_buffer_end;
    size_t out_buffer_end;
};

class CSVWriter {
   public:
    CSVWriter(const boost::filesystem::path& filename,
              const std::vector<std::string>& columns, char _delimiter);

    void add_row(const std::vector<std::string>& columns);

   private:
    ZstdWriter writer;
    size_t num_columns;
    std::string delimiter_str;
    std::string temp;
};

std::vector<std::string> get_csv_columns(ZstdReader& reader, char delimiter);

std::vector<std::string> get_csv_columns(
    const boost::filesystem::path& filename, char delimiter);

class CSVReader {
   public:
    CSVReader(const boost::filesystem::path& filename,
              const std::vector<std::string>& columns, char _delimiter);
    std::vector<std::string>& get_row();

    bool next_row();

   private:
    size_t current_offset;
    ZstdReader reader;
    std::vector<ssize_t> column_map;
    std::vector<std::string> current_row;
    std::deque<bool> current_row_set;
    char delimiter;
};