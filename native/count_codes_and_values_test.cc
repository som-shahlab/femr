
#include "count_codes_and_values.hh"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <boost/filesystem.hpp>
#include <random>

#include "csv.hh"

TEST(CountCodesAndValues, TestCountCodesAndValues) {
    boost::filesystem::path root = boost::filesystem::temp_directory_path() /
                                   boost::filesystem::unique_path();
    boost::filesystem::create_directory(root);

    boost::filesystem::path data_path = root / boost::filesystem::unique_path();
    boost::filesystem::create_directory(data_path);

    std::vector<std::vector<std::string>> entries;

    for (int i = 1; i <= 100; i++) {
        for (int j = 1; j <= 3; j++) {
            entries.push_back({std::to_string(i), std::to_string(j * 100),
                               "a" + std::to_string(i % 3 == 0)});
        }
    }
    // Make sure we ignore numbers
    entries.push_back({"999", "darn", "101"});
    entries.push_back({"999", "darn", "101.1"});
    entries.push_back({"999", "darn", "1e9"});

    entries.push_back({"100", "darn", "please don't waste this"});
    entries.push_back({"100", "darn", "please don't waste this"});

    entries.push_back({"100", "darn", "please don't waste this 2"});

    std::shuffle(std::begin(entries), std::end(entries),
                 std::default_random_engine(1235423));
    std::vector<std::string> columns = {"code", "foo", "value"};

    size_t num_chunks = 7;

    size_t entries_per_chunk = (entries.size() + num_chunks - 1) / num_chunks;

    for (size_t i = 0; i < num_chunks; i++) {
        CSVWriter writer((data_path / std::to_string(i)).string(), columns,
                         ',');
        for (size_t j = 0; j < entries_per_chunk; j++) {
            size_t index = i * entries_per_chunk + j;
            if (index < entries.size()) {
                writer.add_row(entries[index]);
            }
        }
        if (i == 0) {
            writer.add_row({"100", "darn", "edge case"});
            writer.add_row({"100", "darn", "edge case"});
            writer.add_row({"100", "darn", "edge case"});
        }
    }

    boost::filesystem::path tmp_path = root / boost::filesystem::unique_path();

    auto result = count_codes_and_values(data_path, tmp_path, 3);

    for (const auto& entry : result.first) {
        if (entry.first == 100) {
            EXPECT_EQ(entry.second, 9);

        } else {
            EXPECT_EQ(entry.second, 3);
        }
    }

    std::vector<std::pair<std::string, size_t>> value_counts = {
        {"a0", 201},
        {"a1", 99},
        {"edge case", 3},
        {"please don't waste this", 2},
    };

    EXPECT_EQ(result.second, value_counts);

    boost::filesystem::remove_all(root);
}