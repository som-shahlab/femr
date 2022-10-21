#pragma once

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <fstream>

class Dictionary {
   public:
    Dictionary(const boost::filesystem::path& path, bool read_all);
    Dictionary(const Dictionary&) = delete;
    Dictionary(Dictionary&&) = default;

    ~Dictionary() noexcept(false);

    std::string_view operator[](uint32_t idx) const;
    boost::optional<uint32_t> find(std::string_view value);

    const std::vector<std::string_view>& values() const;
    uint32_t size() const;

   private:
    int fd;
    char* mmap_data;
    ssize_t length;

    std::vector<std::string_view> values_;

    const std::vector<uint32_t>& get_sorted_values();
    boost::optional<std::vector<uint32_t>> possib_sorted_values;
};

class DictionaryWriter {
   public:
    DictionaryWriter(const boost::filesystem::path& path);
    ~DictionaryWriter() noexcept(false);

    void add_value(std::string_view value);

   private:
    std::ofstream writer;
    std::vector<uint32_t> entries;
};