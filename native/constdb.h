#ifndef CONSTDB
#define CONSTDB

#include <cstdint>
#include <fstream>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"

class ConstdbReader {
   public:
    ConstdbReader(const char* filename, bool read_all = false,
                  bool old_format = false);
    ~ConstdbReader();

    ConstdbReader(const ConstdbReader&) = delete;
    ConstdbReader& operator=(const ConstdbReader&) = delete;
    ConstdbReader(ConstdbReader&&);

    std::vector<int32_t> get_int_keys() const;

    std::pair<const char*, size_t> get_int(int32_t key) const;
    std::pair<const char*, size_t> get_str(const std::string& key) const;

   private:
    int fd;
    const char* mmap_data;
    absl::flat_hash_map<int32_t, std::pair<const char*, size_t>> int_offsets;
    absl::flat_hash_map<std::string, std::pair<const char*, size_t>>
        str_offsets;
    std::vector<int32_t> int_keys;
};

class ConstdbWriter {
   public:
    ConstdbWriter(const char* filename);
    ~ConstdbWriter();

    void add_int(int32_t key, const char* bytes, int32_t length);
    void add_str(std::string key, const char* bytes, int32_t length);

   private:
    std::ofstream writer;
    std::vector<std::pair<std::variant<int32_t, std::string>, int32_t>> entries;
};

#endif