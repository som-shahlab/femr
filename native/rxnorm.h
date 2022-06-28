#ifndef RXNORM_H_INCLUDED
#define RXNORM_H_INCLUDED

#include <fstream>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/substitute.h"

template <typename F>
void rxnorm_util(std::string_view path, std::string_view name, F f) {
    std::string fname = absl::Substitute("$0/rrf/$1", path, name);

    std::ifstream infile(fname);

    std::string line;
    while (std::getline(infile, line)) {
        std::vector<std::string_view> columns = absl::StrSplit(line, '|');

        f(columns);
    }
}

class RxNorm {
   public:
    RxNorm(std::string_view path) {
        absl::flat_hash_map<std::pair<std::string, std::string>,
                            std::vector<std::string>>
            text_to_rxcui_map;
        rxnorm_util(path, "RXNSAT.RRF", [&](const auto& columns) {
            std::string_view rxcui = columns[0];
            std::string_view atn = columns[8];
            std::string_view atv = columns[10];

            std::string_view actual_sab;

            if (atn == "DHJC") {
                actual_sab = "HCPCS";
            } else if (atn == "NDC") {
                actual_sab = "NDC";
            } else {
                return;
            }

            auto key =
                std::make_pair(std::string(actual_sab), std::string(atv));

            auto [iter, added] = text_to_rxcui_map.insert(
                std::make_pair(key, std::vector<std::string>()));
            (void)added;

            iter->second.push_back(std::string(rxcui));
        });

        for (auto& entry : text_to_rxcui_map) {
            std::sort(std::begin(entry.second), std::end(entry.second));
            entry.second.erase(
                std::unique(std::begin(entry.second), std::end(entry.second)),
                std::end(entry.second));
        }

        absl::flat_hash_set<std::string> valid_relationships = {
            "has_ingredient",        "tradename_of",
            "has_ingredients",       "has_form",
            "consists_of",           "has_part",
            "has_precise_ingredient"};

        absl::flat_hash_map<std::string, std::vector<std::string>> children_map;
        rxnorm_util(path, "RXNREL.RRF", [&](const auto& columns) {
            std::string_view child_cui = columns[0];
            std::string_view parent_cui = columns[4];
            std::string_view rela = columns[7];
            std::string_view sab = columns[10];

            if (sab != "RXNORM") {
                return;
            }

            if (valid_relationships.find(rela) ==
                std::end(valid_relationships)) {
                return;
            }

            auto [iter, added] = children_map.insert(std::make_pair(
                std::string(parent_cui), std::vector<std::string>()));
            (void)added;

            iter->second.push_back(std::string(child_cui));
        });

        absl::flat_hash_map<std::string, std::vector<std::string>>
            all_children_map;
        auto get_all_children = [&](const std::string& rxcui) {
            return get_all_children_helper(rxcui, children_map,
                                           all_children_map);
        };

        std::vector<std::string> rxcuis;

        absl::flat_hash_map<std::string, std::vector<std::string>> atc_map;
        rxnorm_util(path, "RXNCONSO.RRF", [&](const auto& columns) {
            std::string_view cui = columns[0];
            std::string_view sab = columns[11];
            std::string_view code = columns[13];

            if (sab == "RXNORM") {
                rxcuis.push_back(std::string(cui));
            } else if (sab == "ATC") {
                atc_map[std::string(cui)].push_back(std::string(code));
            }
        });
        std::sort(std::begin(rxcuis), std::end(rxcuis));
        rxcuis.erase(std::unique(std::begin(rxcuis), std::end(rxcuis)),
                     std::end(rxcuis));

        for (const auto& entry : text_to_rxcui_map) {
            const auto& key = entry.first;

            std::vector<std::string> atc_codes;
            for (const auto& rxcui : entry.second) {
                for (const auto& child_rxcui : get_all_children(rxcui)) {
                    auto iter = atc_map.find(child_rxcui);
                    if (iter == std::end(atc_map)) {
                        continue;
                    }
                    for (const auto& atc_code : iter->second) {
                        atc_codes.push_back(
                            absl::Substitute("ATC/$0", atc_code));
                    }
                }
            }

            std::sort(std::begin(atc_codes), std::end(atc_codes));
            atc_codes.erase(
                std::unique(std::begin(atc_codes), std::end(atc_codes)),
                std::end(atc_codes));

            text_to_atc_map.insert(std::make_pair(key, atc_codes));
        }

        for (const auto& rxcui : rxcuis) {
            std::vector<std::string> atc_codes;
            for (const auto& child_rxcui : get_all_children(rxcui)) {
                auto iter = atc_map.find(child_rxcui);
                if (iter == std::end(atc_map)) {
                    continue;
                }
                for (const auto& atc_code : iter->second) {
                    atc_codes.push_back(absl::Substitute("ATC/$0", atc_code));
                }
            }

            std::sort(std::begin(atc_codes), std::end(atc_codes));
            atc_codes.erase(
                std::unique(std::begin(atc_codes), std::end(atc_codes)),
                std::end(atc_codes));

            if (atc_codes.size() > 0) {
                text_to_atc_map.insert(
                    std::make_pair(std::make_pair("RxNorm", rxcui), atc_codes));
            }
        }
    }

    std::vector<std::string> get_atc_codes(const std::string& sab,
                                           const std::string& code) const {
        auto iter = text_to_atc_map.find(std::make_pair(sab, code));
        if (iter == std::end(text_to_atc_map)) {
            return {};
        } else {
            return iter->second;
        }
    }

   private:
    absl::flat_hash_map<std::pair<std::string, std::string>,
                        std::vector<std::string>>
        text_to_atc_map;

    std::vector<std::string> get_all_children_helper(
        std::string rxcui,
        const absl::flat_hash_map<std::string, std::vector<std::string>>&
            children_map,
        absl::flat_hash_map<std::string, std::vector<std::string>>&
            all_children_map) {
        auto iter = all_children_map.find(rxcui);
        if (iter == std::end(all_children_map)) {
            std::vector<std::string> result;
            result.push_back(rxcui);

            auto child_iter = children_map.find(rxcui);
            if (child_iter != std::end(children_map)) {
                for (const std::string& child : child_iter->second) {
                    for (const std::string& subchildren :
                         get_all_children_helper(child, children_map,
                                                 all_children_map)) {
                        result.push_back(subchildren);
                    }
                }
            }

            auto p = all_children_map.insert(std::make_pair(rxcui, result));
            iter = p.first;
        }

        return iter->second;
    }
};

#endif