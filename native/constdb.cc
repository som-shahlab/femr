#include "constdb.hh"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

const char* sequential_mmap(int fd, const char* filename, size_t length);

#ifdef POSIX_FADV_SEQUENTIAL
const char* sequential_mmap(int fd, const char* filename, size_t length) {
    int error = posix_fadvise(fd, 0, 0,
                                  POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);

    if (error != 0) {
        printf("Got error trying to set options for %s: %s\n", filename,
               std::strerror(errno));
        exit(-1);
    }

    return (const char*)mmap(nullptr, length, PROT_READ,
                                      MAP_SHARED | MAP_POPULATE, fd, 0);
}
#else
const char* sequential_mmap(int fd, const char* filename, size_t length) {
    return (const char*)mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
}
#endif

ConstdbReader::ConstdbReader(const char* filename, bool read_all) {
    struct stat statbuf;

    if (stat(filename, &statbuf) == -1) {
        printf("Got an error trying to read %s: %s\n", filename,
               std::strerror(errno));
        exit(-1);
    }

    size_t length = statbuf.st_size;

    fd = open(filename, O_RDONLY);
    if (read_all) {
        mmap_data = sequential_mmap(fd, filename, length);
    } else {
        mmap_data =
            (const char*)mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
    }

    uint64_t table_offset = *(uint64_t*)(mmap_data + length - sizeof(uint64_t));

    size_t current_offset = 0;
    uint64_t current_location = table_offset;

    while (current_location < length) {
        int32_t size = *(int32_t*)(mmap_data + current_location);
        current_location += sizeof(int32_t);
        int32_t key_val = *(int32_t*)(mmap_data + current_location);
        current_location += sizeof(int32_t);

        bool is_int;

        if (size < 0) {
            size = -size;
            is_int = false;
        } else {
            is_int = true;
        }

        auto offset =
            std::make_pair((const char*)(mmap_data + current_offset), size);
        current_offset += size;

        if (is_int) {
            int_keys.push_back(key_val);
            int_offsets.insert(std::make_pair(key_val, offset));
        } else {
            std::string key_str =
                std::string((char*)mmap_data + current_location, key_val);
            str_offsets.insert(std::make_pair(key_str, offset));
            current_location += key_val;
        }
    }
}

ConstdbReader::ConstdbReader(ConstdbReader&& other)
    : fd(other.fd),
      mmap_data(other.mmap_data),
      int_offsets(other.int_offsets),
      str_offsets(other.str_offsets),
      int_keys(other.int_keys) {
    other.mmap_data = nullptr;
}

std::vector<int32_t> ConstdbReader::get_int_keys() const {
    if (mmap_data == 0) {
        printf("Invalid reader?\n");
        exit(-1);
    }

    return int_keys;
}

std::pair<const char*, size_t> ConstdbReader::get_int(int32_t key) const {
    if (mmap_data == 0) {
        printf("Invalid reader?\n");
        exit(-1);
    }

    auto iter = int_offsets.find(key);
    if (iter == std::end(int_offsets)) {
        return std::make_pair(nullptr, 0);
    } else {
        return iter->second;
    }
}

std::pair<const char*, size_t> ConstdbReader::get_str(
    const std::string& key) const {
    if (mmap_data == 0) {
        printf("Invalid reader?\n");
        exit(-1);
    }

    auto iter = str_offsets.find(key);
    if (iter == std::end(str_offsets)) {
        return std::make_pair(nullptr, 0);
    } else {
        return iter->second;
    }
}

ConstdbReader::~ConstdbReader() {
    if (mmap_data != 0) {
        close(fd);
    }
}

ConstdbWriter::ConstdbWriter(const char* filename)
    : writer(filename, std::ios_base::out | std::ios_base::binary) {}

void ConstdbWriter::add_int(int32_t key, const char* bytes, int32_t length) {
    writer.write(bytes, length);
    entries.push_back(
        std::make_pair(std::variant<int32_t, std::string>(key), length));
}

void ConstdbWriter::add_str(std::string key, const char* bytes,
                            int32_t length) {
    writer.write(bytes, length);
    entries.push_back(
        std::make_pair(std::variant<int32_t, std::string>(key), length));
}

template <class T>
struct always_false : std::false_type {};

ConstdbWriter::~ConstdbWriter() {
    uint64_t table_offset = writer.tellp();
    for (auto& entry : entries) {
        int32_t length = entry.second;
        std::visit(
            [this, length](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, int32_t>) {
                    int32_t size = length;
                    writer.write((char*)&size, sizeof(size));
                    writer.write((char*)&arg, sizeof(arg));
                } else if constexpr (std::is_same_v<T, std::string>) {
                    int32_t size = -length;
                    writer.write((char*)&size, sizeof(size));
                    int32_t key_length = arg.size();
                    writer.write((char*)&key_length, sizeof(key_length));
                    writer.write(arg.data(), arg.size());
                } else
                    static_assert(always_false<T>::value,
                                  "non-exhaustive visitor!");
            },
            entry.first);
    }
    writer.write((char*)&table_offset, sizeof(table_offset));
}
