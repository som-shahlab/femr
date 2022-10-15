#pragma once

#include <boost/filesystem.hpp>
#include <string>
#include <vector>

void sort_csvs(const boost::filesystem::path& source_directory,
               const boost::filesystem::path& target_directory,
               const std::vector<std::string>& sort_keys, char delimiter,
               size_t num_threads);

void join_csvs(const boost::filesystem::path& source_directory,
               const boost::filesystem::path& target_directory,
               const std::vector<std::string>& sort_keys, char delimiter,
               size_t num_shards);

void sort_and_join_csvs(const boost::filesystem::path& source_directory,
                        const boost::filesystem::path& target_directory,
                        const std::vector<std::string>& sort_keys,
                        char delimiter, size_t num_shards);