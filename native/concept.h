#ifndef CONCEPT_H_INCLUDED
#define CONCEPT_H_INCLUDED

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/substitute.h"
#include "boost/filesystem.hpp"
#include "csv.h"
#include "parse_utils.h"

struct ConceptInfo {
    std::string vocabulary_id;
    std::string concept_code;
    std::string concept_class_id;
};

class ConceptTable {
   public:
    void add_concept(uint32_t concept_id, ConceptInfo info) {
        concepts.insert(std::make_pair(concept_id, info));
    }

    std::optional<ConceptInfo> get_info(uint32_t concept_id) const {
        auto iter = concepts.find(concept_id);

        if (iter == std::end(concepts)) {
            return std::nullopt;
        }

        return iter->second;
    }

    std::optional<uint32_t> get_custom_map(uint32_t concept_id) const {
        auto iter = custom_maps.find(concept_id);
        if (iter != std::end(custom_maps)) {
            return {iter->second};
        } else {
            return {};
        }
    }

    void set_custom_map(uint32_t concept_id, uint32_t maps_to) {
        auto res = custom_maps.insert(std::make_pair(concept_id, maps_to));

        if (!res.second) {
            std::cout << "Got duplicate for " << concept_id << std::endl;
            abort();
        }
    }

   private:
    absl::flat_hash_map<uint32_t, ConceptInfo> concepts;

    absl::flat_hash_map<uint32_t, uint32_t> custom_maps;
};

bool has_prefix(std::string_view a, std::string_view b) {
    return a.substr(0, b.size()) == b;
}

template <typename F>
void file_iter(const std::string& location, std::string_view prefix, F f) {
    std::vector<std::string> options = {
        absl::Substitute("/$0/", prefix),
        absl::Substitute("/$0.csv.gz", prefix),
        absl::Substitute("/$00", prefix),
    };

    for (const auto& target :
         boost::filesystem::recursive_directory_iterator(location)) {
        bool found = false;
        for (const auto& option : options) {
            if (target.path().string().find(option) != std::string::npos &&
                target.path().string().find(".csv.gz") != std::string::npos) {
                found = true;
                break;
            }
        }
        if (found) {
            f(target.path());
        }
    }
}

ConceptTable construct_concept_table(const std::string& location,
                                     char delimiter, bool use_quotes) {
    ConceptTable result;

    std::vector<std::string_view> columns = {
        "concept_id", "vocabulary_id", "concept_code", "concept_class_id"};

    file_iter(location, "concept", [&](const auto& concept_file) {
        csv_iterator(concept_file.c_str(), columns, delimiter, {}, use_quotes,
                     true, [&result](const auto& row) {
                         uint32_t concept_id;
                         attempt_parse_or_die(row[0], concept_id);

                         ConceptInfo info;
                         info.vocabulary_id = row[1];
                         info.concept_code = row[2];
                         info.concept_class_id = row[3];

                         result.add_concept(concept_id, std::move(info));
                     });
    });

    columns = {
        "concept_id_1",
        "concept_id_2",
        "relationship_id",
        "load_table_id",
    };

    file_iter(location, "concept_relationship", [&](const auto& concept_file) {
        csv_iterator(
            concept_file.c_str(), columns, delimiter, {}, use_quotes, true,
            [&result](const auto& row) {
                if (row[2] != "Maps to" || row[3] != "custom_mapping") {
                    return;
                }
                uint32_t concept_id_1, concept_id_2;
                attempt_parse_or_die(row[0], concept_id_1);
                attempt_parse_or_die(row[1], concept_id_2);
                result.set_custom_map(concept_id_1, concept_id_2);
            });
    });

    return result;
}

#endif