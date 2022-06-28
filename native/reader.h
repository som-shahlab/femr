#ifndef READER_H_INCLUDED
#define READER_H_INCLUDED

#include <iostream>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/substitute.h"
#include "constdb.h"
#include "nlohmann/json.hpp"
#include "streamvbyte.h"

class StringHolder {
   public:
    StringHolder(std::string_view a_text) : text(a_text) {}

    StringHolder(const StringHolder& other) {
        if (!other.holder.empty()) {
            holder = other.holder;
            text = holder;
        } else {
            text = other.text;
        }
    }

    StringHolder& operator=(const StringHolder& other) {
        if (!other.holder.empty()) {
            holder = other.holder;
            text = holder;
        } else {
            text = other.text;
        }

        return *this;
    }

    void make_copy() const {
        holder = text;
        text = holder;
    }

    mutable std::string_view text;
    mutable std::string holder;
};

inline bool operator==(const StringHolder& lhs, const StringHolder& rhs) {
    return lhs.text == rhs.text;
}

namespace std {
template <>
struct hash<StringHolder> {
    std::size_t operator()(StringHolder const& s) const noexcept {
        return std::hash<std::string_view>{}(s.text);
    }
};
}  // namespace std

template <typename T>
class Dictionary {
   public:
    explicit Dictionary() { is_mutable = true; }

    explicit Dictionary(const char* begin, size_t size,
                        bool should_be_mutable = false) {
        is_mutable = should_be_mutable;
        nlohmann::json result = nlohmann::json::parse(begin, begin + size);

        reverse_mapper.resize(result["values"].size());

        for (const auto& entry : result["values"]) {
            std::string code_str = entry[0].get<std::string>();
            StringHolder code(code_str);
            code.make_copy();

            uint32_t index = entry[1];
            T count = entry[2].get<T>();
            mapper.emplace(code, index);

            if (index >= reverse_mapper.size()) {
                std::cout << "Values are not contiguous? " << index << " "
                          << reverse_mapper.size() << std::endl;
                abort();
            }

            reverse_mapper[index] = std::make_pair(code_str, count);
        }
    }

    explicit Dictionary(std::vector<std::pair<std::string, T>> vals)
        : reverse_mapper(std::move(vals)) {
        if (reverse_mapper.size() > std::numeric_limits<uint32_t>::max()) {
            std::cout << "Creating a dictionary that's too large "
                      << reverse_mapper.size() << std::endl;
            abort();
        }

        for (size_t i = 0; i < reverse_mapper.size(); i++) {
            StringHolder code(reverse_mapper[i].first);
            code.make_copy();

            mapper.emplace(code, i);
        }
    }

    explicit Dictionary(const Dictionary& source)
        : Dictionary(source.reverse_mapper) {}

    std::optional<uint32_t> map(std::string_view code) const {
        auto iter = mapper.find(StringHolder(code));

        if (iter == std::end(mapper)) {
            return {};
        } else {
            return {iter->second};
        }
    }

    std::optional<std::string_view> get_word(uint32_t code) const {
        if (code >= reverse_mapper.size()) {
            return {};
        } else {
            return {reverse_mapper[code].first};
        }
    }

    void clear() {
        if (!is_mutable) {
            std::cout << absl::Substitute("Trying to clear an immutable map\n");
            abort();
        }

        mapper.clear();
        reverse_mapper.clear();
    }

    std::string to_json() const {
        std::vector<std::tuple<std::string, uint32_t, T>> entries;

        for (size_t i = 0; i < reverse_mapper.size(); i++) {
            const auto& entry = reverse_mapper[i];
            entries.push_back(std::make_tuple(entry.first, i, entry.second));
        }

        nlohmann::json result;
        result["values"] = entries;

        return result.dump();
    }

    std::vector<std::pair<std::string, uint32_t>> decompose() const {
        return reverse_mapper;
    }

    uint32_t size() const { return reverse_mapper.size(); }

   protected:
    absl::flat_hash_map<StringHolder, uint32_t> mapper;
    std::vector<std::pair<std::string, T>> reverse_mapper;
    bool is_mutable;
};

class TermDictionary : public Dictionary<uint32_t> {
   public:
    using Dictionary::Dictionary;

    uint32_t map_or_add(std::string_view code, uint32_t count = 1) {
        auto [iter, added] =
            mapper.emplace(StringHolder(code), reverse_mapper.size());
        if (added) {
            if (!is_mutable) {
                std::cout << absl::Substitute(
                    "Trying to add an item to an immutable map\n");
                abort();
            }
            reverse_mapper.push_back(std::make_pair(std::string(code), 0));
            iter->first.make_copy();

            if (reverse_mapper.size() > std::numeric_limits<uint32_t>::max()) {
                std::cout << "Adding a word to a dictionary that is too large "
                          << reverse_mapper.size() << std::endl;
                abort();
            }
        }

        uint32_t& total = reverse_mapper[iter->second].second;

        uint64_t temp_total = total;
        temp_total += count;

        if (temp_total > std::numeric_limits<uint32_t>::max()) {
            std::cout << absl::Substitute("Hit max for code $0 $1\n", code,
                                          temp_total);
            abort();
        }

        total = temp_total;

        return iter->second;
    }

    std::pair<TermDictionary, std::vector<uint32_t>> optimize() const {
        std::vector<std::tuple<int32_t, uint32_t, std::string>> entries;

        for (size_t i = 0; i < reverse_mapper.size(); i++) {
            const auto& entry = reverse_mapper[i];
            entries.push_back(std::make_tuple(-entry.second, i, entry.first));
        }

        std::sort(std::begin(entries), std::end(entries));

        std::vector<uint32_t> mapper_vec(entries.size());
        std::vector<std::pair<std::string, uint32_t>> new_reverse_mapper;

        for (size_t i = 0; i < entries.size(); i++) {
            const auto& entry = entries[i];
            mapper_vec[std::get<1>(entry)] = i;
            new_reverse_mapper.push_back(
                std::make_pair(std::get<2>(entry), -std::get<1>(entry)));
        }

        return std::make_pair(TermDictionary(new_reverse_mapper), mapper_vec);
    }
};

class OntologyCodeDictionary : public Dictionary<std::string> {
   public:
    using Dictionary::Dictionary;

    uint32_t add(std::string_view code, std::string_view definition) {
        auto [iter, added] =
            mapper.emplace(StringHolder(code), reverse_mapper.size());

        if (added) {
            reverse_mapper.push_back(
                std::make_pair(std::string(code), std::string(definition)));
            iter->first.make_copy();

            if (reverse_mapper.size() > std::numeric_limits<uint32_t>::max()) {
                std::cout << "Adding a word to a dictionary that is too large "
                          << reverse_mapper.size() << std::endl;
                abort();
            }
        } else {
            std::cout << "Got duplicate definition for word " << code
                      << "Existing: " << reverse_mapper[iter->second].second
                      << "New: " << definition << std::endl;
            abort();
        }
        return iter->second;
    }

    std::optional<std::string_view> get_definition(uint32_t code) const {
        if (code >= reverse_mapper.size()) {
            return {};
        } else {
            return {reverse_mapper[code].second};
        }
    }
};

template <class To, class From>
typename std::enable_if<
    (sizeof(To) == sizeof(From)) && std::is_trivially_copyable<From>::value &&
        std::is_trivial<To>::value,
    // this implementation requires that To is trivially default constructible
    To>::type
// constexpr support needs compiler magic
bit_cast(const From& src) noexcept {
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

struct ObservationWithValue {
    ObservationWithValue() {}

    ObservationWithValue(uint32_t source_code, uint32_t value) {
        code = source_code >> 2;
        uint32_t code_type = source_code & 3;

        if (code_type == 2) {
            is_text = true;
            text_value = value;
        } else if (code_type == 0) {
            is_text = false;
            numeric_value = value;
        } else if (code_type == 1) {
            is_text = false;
            numeric_value = bit_cast<float>(value);
        } else {
            std::cout << "Invalid code type?" << code_type << std::endl;
            abort();
        }
    }

    std::pair<uint32_t, uint32_t> encode() const {
        uint32_t final_code = code << 2;

        uint32_t code_type;
        uint32_t final_value;

        if (is_text) {
            code_type = 2;
            final_value = text_value;
        } else {
            uint32_t numeric_value_as_int = (uint32_t)numeric_value;
            float round_trip = (float)numeric_value_as_int;
            if (round_trip == numeric_value) {
                code_type = 0;
                final_value = numeric_value_as_int;
            } else {
                code_type = 1;
                final_value = bit_cast<uint32_t>(numeric_value);
            }
        }

        return std::make_pair(final_code + code_type, final_value);
    }

    uint32_t code;
    bool is_text;

    union {
        uint32_t text_value;
        float numeric_value;
    };
};

class ExtractReaderIterator {
   public:
    ExtractReaderIterator(const ConstdbReader* r) : reader(r) {}
    ExtractReaderIterator(const ExtractReaderIterator& other)
        : reader(other.reader) {}

    template <typename F>
    bool process_patient(uint32_t patient_id, F f) {
        buffer.clear();

        auto [patient_data, num_bytes] = reader->get_int(patient_id);

        if (patient_data == nullptr) {
            return false;
        }

        const uint32_t* size = (const uint32_t*)patient_data;
        const uint8_t* main_data =
            (const uint8_t*)patient_data + sizeof(uint32_t);

        if (*size > buffer.size()) {
            buffer.resize(*size * 2 + 1);
        }

        size_t bytes_read = streamvbyte_decode(main_data, buffer.data(), *size);

        if (bytes_read + sizeof(uint32_t) != num_bytes) {
            std::cout << absl::Substitute("Invalid parsing? $0 $1 $2\n",
                                          patient_id, bytes_read, num_bytes);
            abort();
        }

        size_t index = 0;

        uint32_t year = buffer[index++];
        uint32_t month = buffer[index++];
        uint32_t day = buffer[index++];

        absl::CivilDay birth_day(year, month, day);

        uint32_t num_days = buffer[index++];
        uint32_t age = 0;

        for (uint32_t i = 0; i < num_days; i++) {
            observations.clear();
            observations_with_values.clear();

            uint32_t delta = buffer[index++];
            age += delta;

            uint32_t num_obs = buffer[index++];

            uint32_t current_obs = 0;

            for (uint32_t obs_i = 0; obs_i < num_obs; obs_i++) {
                current_obs += buffer[index++];
                observations.push_back(current_obs);
            }

            uint32_t num_obs_with_values = buffer[index++];

            current_obs = 0;
            for (uint32_t obs_i = 0; obs_i < num_obs_with_values; obs_i++) {
                current_obs += buffer[index++];
                uint32_t value = buffer[index++];
                observations_with_values.push_back(
                    ObservationWithValue(current_obs, value));
            }

            f(birth_day, age, observations, observations_with_values);
        }

        return true;
    }

   private:
    const ConstdbReader* reader;

    std::vector<uint32_t> buffer;
    std::vector<uint32_t> observations;
    std::vector<ObservationWithValue> observations_with_values;
};

class ExtractReader {
   public:
    ExtractReader(const char* path, bool read_all) : reader(path, read_all) {}

    absl::Span<const uint32_t> get_patient_ids() const {
        const uint32_t* num_patients =
            (const uint32_t*)reader.get_str("num_patients").first;
        const uint32_t* patient_ids_ptr =
            (const uint32_t*)reader.get_str("patient_ids").first;
        return absl::Span<const uint32_t>(patient_ids_ptr, *num_patients);
    }

    absl::Span<const uint64_t> get_original_patient_ids() const {
        const uint32_t* num_patients =
            (const uint32_t*)reader.get_str("num_patients").first;
        const uint64_t* original_patient_ids_ptr =
            (const uint64_t*)reader.get_str("original_ids").first;
        return absl::Span<const uint64_t>(original_patient_ids_ptr,
                                          *num_patients);
    }

    ExtractReaderIterator iter() const {
        return ExtractReaderIterator(&reader);
    }

    const TermDictionary& get_dictionary() {
        if (!dictionary) {
            auto [dict_start, dict_size] = reader.get_str("dictionary");
            dictionary = TermDictionary(dict_start, dict_size);
        }

        return *dictionary;
    }

    const TermDictionary& get_value_dictionary() {
        if (!value_dictionary) {
            auto [val_dict_start, val_dict_size] =
                reader.get_str("value_dictionary");
            value_dictionary = TermDictionary(val_dict_start, val_dict_size);
        }

        return *value_dictionary;
    }

   private:
    ConstdbReader reader;

    std::optional<TermDictionary> dictionary;
    std::optional<TermDictionary> value_dictionary;
};

class OntologyReader {
   public:
    OntologyReader(const char* path) : reader(path, false) {
        for (uint32_t subword = 0; subword < get_dictionary().size();
             subword++) {
            for (uint32_t parent : get_parents(subword)) {
                children_map[parent].push_back(subword);
            }
        }

        for (auto& item : children_map) {
            std::sort(std::begin(item.second), std::end(item.second));
        }
        auto [ptr, size] = reader.get_str("root");
        if (size != sizeof(uint32_t)) {
            std::cout << "Could not find the root code " << std::endl;
            abort();
        }
        const uint32_t* root_code_ptr = (const uint32_t*)ptr;
        root_code = *root_code_ptr;
    }

    absl::Span<const uint32_t> get_subwords(uint32_t code) const {
        auto [ptr, size] = reader.get_int(code);
        return absl::Span<const uint32_t>((const uint32_t*)ptr,
                                          size / sizeof(uint32_t));
    }

    absl::Span<const uint32_t> get_parents(uint32_t subword) const {
        int32_t subword_copy = subword + 1;
        auto [ptr, size] = reader.get_int(-subword_copy);
        return absl::Span<const uint32_t>((const uint32_t*)ptr,
                                          size / sizeof(uint32_t));
    }

    absl::Span<const uint32_t> get_all_parents(uint32_t subword) {
        auto iter = all_parents_map.find(subword);

        if (iter == std::end(all_parents_map)) {
            std::vector<uint32_t> parents;
            for (uint32_t parent : get_parents(subword)) {
                parents.push_back(parent);

                for (uint32_t parent_parent : get_all_parents(parent)) {
                    parents.push_back(parent_parent);
                }
            }

            std::sort(std::begin(parents), std::end(parents));
            parents.erase(std::unique(std::begin(parents), std::end(parents)),
                          std::end(parents));

            auto res = all_parents_map.insert(
                std::make_pair(subword, std::move(parents)));
            iter = res.first;
        }

        return iter->second;
    }

    absl::Span<const uint32_t> get_children(uint32_t subword) const {
        auto iter = children_map.find(subword);
        if (iter == std::end(children_map)) {
            return {};
        } else {
            return iter->second;
        }
    }

    absl::Span<const uint32_t> get_words_for_subword(uint32_t subword_id) {
        const auto& inverse_map = get_inverse_map();

        auto iter = inverse_map.find(subword_id);
        if (iter == std::end(inverse_map)) {
            return {};
        } else {
            return iter->second;
        }
    }

    absl::Span<const uint32_t> get_words_for_subword_term(
        std::string_view subword_term) {
        std::optional<uint32_t> code = get_dictionary().map(subword_term);

        if (!code) {
            return {};
        } else {
            return get_words_for_subword(*code);
        }
    }

    const TermDictionary& get_dictionary() {
        if (!dictionary) {
            auto [dict_start, dict_size] = reader.get_str("dictionary");
            dictionary = TermDictionary(dict_start, dict_size);
        }

        return *dictionary;
    }

    absl::Span<const uint32_t> get_recorded_date_codes() const {
        auto [ptr, size] = reader.get_str("recorded_date_codes");
        absl::Span<const uint32_t> recorded_date_codes((const uint32_t*)ptr,
                                                       size / sizeof(uint32_t));
        return recorded_date_codes;
    }

    const OntologyCodeDictionary& get_text_description_dictionary() {
        if (!text_description_dictionary) {
            auto [dict_start, dict_size] =
                reader.get_str("text_description_dictionary");
            std::cout << "Got text description dictionary of size " << dict_size
                      << std::endl;
            text_description_dictionary =
                OntologyCodeDictionary(dict_start, dict_size);
        }

        return *text_description_dictionary;
    }

    uint32_t get_root_code() const { return root_code; }

   private:
    const absl::flat_hash_map<uint32_t, std::vector<uint32_t>>&
    get_inverse_map() {
        if (!inverse_map) {
            absl::flat_hash_map<uint32_t, std::vector<uint32_t>> result;
            auto [ptr, size] = reader.get_str("words_with_subwords");
            absl::Span<const uint32_t> words_with_subwords(
                (const uint32_t*)ptr, size / sizeof(uint32_t));
            for (uint32_t word : words_with_subwords) {
                for (uint32_t subword : get_subwords(word)) {
                    result[subword].push_back(word);
                }
            }

            for (auto& item : result) {
                std::sort(std::begin(item.second), std::end(item.second));
            }

            inverse_map = std::move(result);
        }

        return *inverse_map;
    }

    uint32_t root_code;

    ConstdbReader reader;

    std::optional<TermDictionary> dictionary;

    std::optional<OntologyCodeDictionary> text_description_dictionary;

    std::optional<absl::flat_hash_map<uint32_t, std::vector<uint32_t>>>
        inverse_map;

    absl::flat_hash_map<uint32_t, std::vector<uint32_t>> children_map;
    std::map<uint32_t, std::vector<uint32_t>> all_parents_map;
};

class Index {
   public:
    Index(const char* filename) : reader(filename, false) {}

    std::vector<uint32_t> get_patient_ids(uint32_t term) {
        auto [iter_ptr, iter_size] = reader.get_int(term);
        if (iter_ptr == nullptr) {
            return {};
        }

        const uint32_t* size_ptr = (const uint32_t*)iter_ptr;
        const uint8_t* main_data =
            (const uint8_t*)(iter_ptr + sizeof(uint32_t));

        std::vector<uint32_t> result(*size_ptr);

        size_t bytes_read =
            streamvbyte_decode(main_data, result.data(), *size_ptr);

        if (bytes_read + sizeof(uint32_t) != iter_size) {
            std::cout << "Invalid compressed data?" << std::endl;
            abort();
        }

        uint32_t last_pid = 0;

        for (uint32_t& pid : result) {
            pid += last_pid;
            last_pid = pid;
        }

        return result;
    }

    std::vector<uint32_t> get_all_patient_ids(
        const std::vector<uint32_t>& terms) {
        std::vector<uint32_t> final_result;

        for (uint32_t term : terms) {
            std::vector<uint32_t> result = get_patient_ids(term);
            final_result.insert(std::end(final_result), std::begin(result),
                                std::end(result));
        }

        std::sort(std::begin(final_result), std::end(final_result));
        final_result.erase(
            std::unique(std::begin(final_result), std::end(final_result)),
            std::end(final_result));

        return final_result;
    }

   private:
    ConstdbReader reader;
};

#endif
