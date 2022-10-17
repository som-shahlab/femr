#pragma once

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <fstream>

class Dictionary {
   public:
    Dictionary(const boost::filesystem::path& path, bool read_all);
    ~Dictionary() noexcept(false);

    std::string_view get_text(uint32_t code) const;
    boost::optional<uint32_t> get_code(std::string_view word);

    const std::vector<std::string_view>& get_all_text() const;

    uint32_t get_num_entries() const;

   private:
    int fd;
    char* mmap_data;
    ssize_t length;

    std::vector<std::string_view> values;

    const std::vector<uint32_t>& get_sorted_values();
    boost::optional<std::vector<uint32_t>> possib_sorted_values;
};

class DictionaryWriter {
   public:
    DictionaryWriter(const boost::filesystem::path& path);
    ~DictionaryWriter() noexcept(false);

    void add_value(std::string_view value);

   private:
    uint64_t total_size;
    std::ofstream writer;
    std::vector<uint32_t> entries;
};