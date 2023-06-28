#pragma once

#include <boost/filesystem.hpp>
#include <string>
#include <vector>

std::pair<std::vector<std::pair<int64_t, size_t>>,
          std::vector<std::pair<std::string, size_t>>>
count_codes_and_values(const boost::filesystem::path& path,
                       const boost::filesystem::path& tmp_path,
                       size_t num_threads);
