
#include "join_csvs.hh"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <random>

#include "csv.hh"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

TEST(JoinCsvTest, TestSort) {
    boost::filesystem::path root = boost::filesystem::temp_directory_path() /
                                   boost::filesystem::unique_path();
    boost::filesystem::create_directory(root);
    boost::filesystem::path source = root / "sorted_source_dir";
    boost::filesystem::path target = root / "sorted_target_dir";
    boost::filesystem::create_directory(source);
    std::vector<std::string> columns = {"col1", "col2"};

    std::vector<std::vector<std::string>> entries;

    for (int i = 1; i <= 100; i++) {
        for (int j = 1; j <= 3; j++) {
            entries.push_back({std::to_string(i), std::to_string(j * 100)});
        }
    }

    std::shuffle(std::begin(entries), std::end(entries),
                 std::default_random_engine(1235423));

    size_t num_chunks = 7;

    size_t entries_per_chunk = (entries.size() + num_chunks - 1) / num_chunks;

    for (size_t i = 0; i < num_chunks; i++) {
        CSVWriter writer((source / std::to_string(i)).string(), columns, ',');
        for (size_t j = 0; j < entries_per_chunk; j++) {
            size_t index = i * entries_per_chunk + j;
            if (index < entries.size()) {
                writer.add_row(entries[index]);
            }
        }
    }

    sort_csvs(source.string(), target.string(), {"col1", "col2"}, ',', 5);

    for (size_t i = 0; i < num_chunks; i++) {
        std::sort(std::begin(entries) + entries_per_chunk * i,
                  std::begin(entries) +
                      std::min(entries.size(), entries_per_chunk * (i + 1)));
        CSVReader reader((target / std::to_string(i)).string(), columns, ',');
        for (size_t j = 0; j < entries_per_chunk; j++) {
            size_t index = i * entries_per_chunk + j;
            if (index < entries.size()) {
                EXPECT_EQ(reader.next_row(), true);
                EXPECT_EQ(reader.get_row(), entries[index]);
            }
        }
        EXPECT_EQ(reader.next_row(), false);
    }
    boost::filesystem::remove_all(root);
}

TEST(JoinCsvTest, TestSortAndJoin) {
    boost::filesystem::path root = boost::filesystem::temp_directory_path() /
                                   boost::filesystem::unique_path();
    boost::filesystem::create_directory(root);
    boost::filesystem::path source = root / "source_dir";
    boost::filesystem::path target = root / "target_dir";
    boost::filesystem::create_directory(source);
    std::vector<std::string> columns = {"col1", "col2"};

    std::vector<std::vector<std::string>> entries;

    for (int i = 1; i <= 100; i++) {
        for (int j = 1; j <= 3; j++) {
            entries.push_back({std::to_string(i), std::to_string(j * 100)});
        }
    }

    std::shuffle(std::begin(entries), std::end(entries),
                 std::default_random_engine(1235423));

    size_t num_chunks = 7;

    size_t entries_per_chunk = (entries.size() + num_chunks - 1) / num_chunks;

    size_t num_shards = 5;

    for (size_t i = 0; i < num_chunks; i++) {
        CSVWriter writer((source / std::to_string(i)).string(), columns, ',');
        for (size_t j = 0; j < entries_per_chunk; j++) {
            size_t index = i * entries_per_chunk + j;
            if (index < entries.size()) {
                writer.add_row(entries[index]);
            }
        }
    }

    std::vector<std::string> additional_sort = {"col2"};
    sort_and_join_csvs(source.string(), target.string(), {"col1", "col2"}, ',',
                       num_shards);

    int num_keys = 0;
    int found_shards = 0;
    for (auto& entry : boost::make_iterator_range(
             boost::filesystem::directory_iterator(target), {})) {
        found_shards += 1;
        CSVReader reader(entry, columns, ',');
        while (reader.next_row()) {
            num_keys += 1;
            std::string key = reader.get_row()[0];
            EXPECT_EQ(reader.get_row()[0], key);
            EXPECT_EQ(reader.get_row()[1], "100");
            EXPECT_EQ(reader.next_row(), true);
            EXPECT_EQ(reader.get_row()[0], key);
            EXPECT_EQ(reader.get_row()[1], "200");
            EXPECT_EQ(reader.next_row(), true);
            EXPECT_EQ(reader.get_row()[0], key);
            EXPECT_EQ(reader.get_row()[1], "300");
        }
    }

    EXPECT_EQ(num_keys, 100);
    EXPECT_EQ(found_shards, 5);
    boost::filesystem::remove_all(root);
}