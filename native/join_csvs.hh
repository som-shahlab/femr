#pragma once

#include <boost/filesystem.hpp>
#include <string>
#include <vector>

enum class ColumnValueType { STRING, UINT64_T, INT64_T, DATETIME };

void sort_csvs(
    const boost::filesystem::path& source_directory,
    const boost::filesystem::path& target_directory,
    const std::vector<std::pair<std::string, ColumnValueType>>& sort_keys,
    char delimiter, size_t num_threads);

void join_csvs(
    const boost::filesystem::path& source_directory,
    const boost::filesystem::path& target_directory,
    const std::vector<std::pair<std::string, ColumnValueType>>& sort_keys,
    char delimiter, size_t num_shards);

void sort_and_join_csvs(
    const boost::filesystem::path& source_directory,
    const boost::filesystem::path& target_directory,
    const std::vector<std::pair<std::string, ColumnValueType>>& sort_keys,
    char delimiter, size_t num_shards);
