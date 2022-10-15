#include "dictionary.hh"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "absl/strings/str_cat.h"
#include "streamvbyte.h"

char* setup_read_all(int fd, size_t length) {
#ifdef MAP_POPULATE
    int error =
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);

    if (error < 0) {
        throw std::runtime_error(absl::StrCat(
            "Got error trying to set options for ", std::strerror(errno)));
    }

    return static_cast<char*>(
        mmap(nullptr, length, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0));
#else
    return static_cast<char*>(
        mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0));
#endif
}

Dictionary::Dictionary(const boost::filesystem::path& path, bool read_all) {
    if (!boost::filesystem::exists(path)) {
        throw std::runtime_error("Could not find file " + path.string());
    }

    {
        uintmax_t file_size = boost::filesystem::file_size(path);

        if (file_size > std::numeric_limits<ssize_t>::max()) {
            throw std::runtime_error(absl::StrCat(
                "Cannot map file larger than ssize_t::max ", path.string()));
        }
        length = file_size;
    }

    fd = open(path.c_str(), O_RDONLY);

    if (fd < 0) {
        throw std::runtime_error(absl::StrCat("Got error trying to open for ",
                                              path.string(), " ",
                                              std::strerror(errno)));
    }

    if (read_all) {
        mmap_data = setup_read_all(fd, length + STREAMVBYTE_PADDING);

    } else {
        mmap_data =
            static_cast<char*>(mmap(nullptr, length + STREAMVBYTE_PADDING,
                                    PROT_READ, MAP_SHARED, fd, 0));
    }

    if (mmap_data == reinterpret_cast<const char*>(-1)) {
        throw std::runtime_error(absl::StrCat("Got error trying to mmap ",
                                              path.string(), " ",
                                              std::strerror(errno)));
    }

    uint64_t table_offset =
        *reinterpret_cast<uint64_t*>(mmap_data + (length - sizeof(uint64_t)));

    uint32_t num_sizes = *reinterpret_cast<uint32_t*>(mmap_data + table_offset);
    std::vector<uint32_t> sizes(num_sizes);
    streamvbyte_decode(reinterpret_cast<const uint8_t*>(
                           mmap_data + table_offset + sizeof(uint32_t)),
                       sizes.data(), num_sizes);

    uint64_t index = 0;
    for (const auto& size : sizes) {
        values.emplace_back(mmap_data + index, size);
        index += size;
    }
}

Dictionary::~Dictionary() noexcept(false) {
    int ret = munmap(mmap_data, length);
    if (ret < 0) {
        throw std::runtime_error(absl::StrCat(
            "Got error trying to unmap Dictionary", std::strerror(errno)));
    }
    ret = close(fd);
    if (ret < 0) {
        throw std::runtime_error(absl::StrCat(
            "Got error trying to close Dictionary", std::strerror(errno)));
    }
}

uint32_t Dictionary::get_num_entries() const { return values.size(); }

std::string_view Dictionary::get_text(uint32_t code) const {
    if (code >= values.size()) {
        throw std::runtime_error(absl::StrCat("Cannot look up index ", code,
                                              " in dictionary of size ",
                                              values.size()));
    }
    return values[code];
}

std::optional<uint32_t> Dictionary::get_code(std::string_view word) {
    const auto& sorted = get_sorted_values();
    auto iter = std::lower_bound(
        std::begin(sorted), std::end(sorted), word,
        [&](uint32_t a, std::string_view b) { return values[a] < b; });

    if (iter == std::end(sorted) || values[*iter] != word) {
        return std::nullopt;
    } else {
        return *iter;
    }
}

const std::vector<std::string_view>& Dictionary::get_all_text() const {
    return values;
}

const std::vector<uint32_t>& Dictionary::get_sorted_values() {
    if (!possib_sorted_values) {
        possib_sorted_values.emplace();
        possib_sorted_values->reserve(values.size());
        for (size_t i = 0; i < values.size(); i++) {
            possib_sorted_values->push_back(i);
        }

        std::sort(
            std::begin(*possib_sorted_values), std::end(*possib_sorted_values),
            [&](uint32_t a, uint32_t b) { return values[a] < values[b]; });
    }
    return *possib_sorted_values;
}

DictionaryWriter::DictionaryWriter(const boost::filesystem::path& path)
    : writer(path.string()) {}

DictionaryWriter::~DictionaryWriter() noexcept(false) {
    std::vector<char> compressed(
        streamvbyte_max_compressedbytes(entries.size()));

    size_t compressed_size =
        streamvbyte_encode(entries.data(), entries.size(),
                           reinterpret_cast<uint8_t*>(compressed.data()));

    if (writer.tellp() > std::numeric_limits<int64_t>::max()) {
        throw std::runtime_error(
            "Cannot work withs files greaterthan uint64_t::max");
    }

    uint32_t num_entries = entries.size();
    uint64_t position = writer.tellp();

    writer.write(reinterpret_cast<const char*>(&num_entries),
                 sizeof(num_entries));
    writer.write(compressed.data(), compressed_size);
    writer.write(reinterpret_cast<const char*>(&position), sizeof(position));
}

void DictionaryWriter::add_value(std::string_view value) {
    if (value.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(
            "Cannot store values of size greater than uint32_t::max");
    }
    writer.write(value.data(), value.size());
    entries.push_back(value.size());

    if (entries.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Cannot store more values than uint32_t::max");
    }
}