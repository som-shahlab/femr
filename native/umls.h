#ifndef UMLS_H_INCLUDED
#define UMLS_H_INCLUDED

#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/substitute.h"

class UMLS {
   public:
    UMLS(std::string umls_path) {
        code_to_aui_map = load_code_to_aui_map(umls_path);
        for (auto& iter : code_to_aui_map) {
            aui_to_code_map.insert(std::make_pair(iter.second, iter.first));
        }

        aui_to_parents_map =
            load_aui_to_parents_map(umls_path, aui_to_code_map);

        aui_to_text_description_map =
            aui_to_description_map(umls_path, code_to_aui_map);
    }

    std::optional<std::string> get_aui(const std::string& sab,
                                       const std::string& code) const {
        auto iter = code_to_aui_map.find(std::make_pair(sab, code));
        if (iter == std::end(code_to_aui_map)) {
            return std::nullopt;
        } else {
            return {iter->second};
        }
    }

    std::optional<std::pair<std::string, std::string>> get_code(
        const std::string& aui) const {
        auto iter = aui_to_code_map.find(aui);
        if (iter == std::end(aui_to_code_map)) {
            return std::nullopt;
        } else {
            return {iter->second};
        }
    }

    std::vector<std::string> get_parents(const std::string& aui) const {
        auto iter = aui_to_parents_map.find(aui);
        if (iter == std::end(aui_to_parents_map)) {
            return {};
        } else {
            return iter->second;
        }
    }

    std::optional<std::string> get_definition(const std::string& aui) const {
        auto iter = aui_to_text_description_map.find(aui);
        if (iter == std::end(aui_to_text_description_map)) {
            return std::nullopt;
        } else {
            return {iter->second};
        }
    }

   private:
    absl::flat_hash_map<std::pair<std::string, std::string>, std::string>
        code_to_aui_map;
    absl::flat_hash_map<std::string, std::pair<std::string, std::string>>
        aui_to_code_map;
    absl::flat_hash_map<std::string, std::vector<std::string>>
        aui_to_parents_map;
    absl::flat_hash_map<std::string, std::string> aui_to_text_description_map;

    absl::flat_hash_map<std::pair<std::string, std::string>, std::string>
    load_code_to_aui_map(std::string umls_path) {
        std::string mrconso =
            absl::Substitute("$0/$1", umls_path, "MRCONSO.RRF");

        if (!boost::filesystem::exists(mrconso)) {
            std::cout << "Could not find the MRCONSO.RRF file in the provided "
                         "umls_path "
                      << umls_path << std::endl;
            exit(-1);
        }

        std::ifstream infile(mrconso);

        absl::flat_hash_map<std::pair<std::string, std::string>, std::string>
            result;

        std::set<std::string> main_ttys = {"PT",  "HT",  "LN", "IN",
                                           "RHT", "LPN", "OP", "LO"};

        std::set<std::string> icd10pcs_ttys = {
            "HX",
            "PX",
            "MTH_HX",
        };

        std::set<std::string> valid_sabs = {"HCPCS",   "CPT",     "SRC",
                                            "ICD10CM", "ATC",     "LNC",
                                            "MTHHH",   "ICD10PCS"};

        std::string line;
        while (std::getline(infile, line)) {
            std::vector<std::string_view> columns = absl::StrSplit(line, '|');

            std::string code(columns[13]);
            std::string sab(columns[11]);
            std::string aui(columns[7]);
            std::string tty(columns[12]);

            std::set<std::string>* valid_ttys;
            if (sab == "ICD10PCS") {
                valid_ttys = &icd10pcs_ttys;
            } else {
                valid_ttys = &main_ttys;
            }

            if (aui == "A1415709" ||
                (valid_ttys->find(tty) != std::end(*valid_ttys) &&
                 valid_sabs.find(sab) != std::end(valid_sabs))) {
                auto [iter, added] = result.insert(
                    std::make_pair(std::make_pair(sab, code), aui));
                if (!added) {
                    std::cout << "Got duplicate aui " << code << " " << sab
                              << " " << aui << " " << iter->second << std::endl;
                    abort();
                }
            }
        }

        return result;
    }

    absl::flat_hash_map<std::string, std::string> load_atc_fix_map(
        std::string umls_path) {
        std::string mrconso =
            absl::Substitute("$0/$1", umls_path, "MRCONSO.RRF");

        std::ifstream infile(mrconso);

        absl::flat_hash_map<std::string, std::string> atc_pt;
        absl::flat_hash_map<std::string, std::string> atc_rxn_pt;

        std::string line;
        while (std::getline(infile, line)) {
            std::vector<std::string_view> columns = absl::StrSplit(line, '|');

            std::string code(columns[13]);
            std::string sab(columns[11]);
            std::string aui(columns[7]);
            std::string tty(columns[12]);

            if (sab == "ATC") {
                if (tty == "PT" || tty == "IN") {
                    atc_pt[code] = aui;
                } else if (tty == "RXN_PT" || tty == "RXN_IN") {
                    atc_rxn_pt[aui] = code;
                }
            } else if (sab == "ICD10PCS") {
                if (tty == "PT" || tty == "HT") {
                    atc_rxn_pt[aui] = code;
                } else if (tty == "PX" || tty == "HX") {
                    atc_pt[code] = aui;
                }
            }
        }

        absl::flat_hash_map<std::string, std::string> result;

        for (auto& entry : atc_rxn_pt) {
            auto iter = atc_pt.find(entry.second);
            if (iter == std::end(atc_pt)) {
                std::cout << "Could not find actual value for " << entry.second
                          << std::endl;
                abort();
            }
            result[entry.first] = iter->second;
        }

        return result;
    }

    absl::flat_hash_map<std::string, std::vector<std::string>>
    load_aui_to_parents_map(
        std::string umls_path,
        const absl::flat_hash_map<std::string,
                                  std::pair<std::string, std::string>>&
            aui_to_code_map) {
        auto atc_fix_map = load_atc_fix_map(umls_path);

        std::string mrrel = absl::Substitute("$0/$1", umls_path, "MRREL.RRF");
        if (!boost::filesystem::exists(mrrel)) {
            std::cout << "Could not find the MRREL.RRF file in the provided "
                         "umls_path "
                      << umls_path << std::endl;
            exit(-1);
        }

        std::ifstream infile(mrrel);

        absl::flat_hash_map<std::string, std::vector<std::string>> result;

        std::string line;
        while (std::getline(infile, line)) {
            std::vector<std::string_view> columns = absl::StrSplit(line, '|');

            std::string aui1(columns[1]);
            std::string aui2(columns[5]);
            std::string rel(columns[3]);

            if (rel != "PAR") {
                continue;
            }

            auto fixed1 = atc_fix_map.find(aui1);
            if (fixed1 != std::end(atc_fix_map)) {
                aui1 = fixed1->second;
            }

            auto fixed2 = atc_fix_map.find(aui2);
            if (fixed2 != std::end(atc_fix_map)) {
                aui2 = fixed2->second;
            }

            if (aui_to_code_map.find(aui1) == std::end(aui_to_code_map) ||
                aui_to_code_map.find(aui2) == std::end(aui_to_code_map)) {
                continue;
            }

            result[aui1].push_back(aui2);
        }

        result["A6321000"].push_back("A1415709");
        result["A23576389"].push_back("A1415709");
        result["A20098492"].push_back("A1415709");
        result["A22725500"].push_back("A1415709");
        result["A13475665"].push_back("A1415709");
        result["A16077350"].push_back("A1415709");

        return result;
    }

    absl::flat_hash_map<std::string, std::string> aui_to_description_map(
        std::string umls_path,
        const absl::flat_hash_map<std::pair<std::string, std::string>,
                                  std::string>& code_to_aui_map) {
        std::string mrconso =
            absl::Substitute("$0/$1", umls_path, "MRCONSO.RRF");

        std::ifstream infile(mrconso);

        absl::flat_hash_map<std::string, std::string> result;

        std::string line;
        while (std::getline(infile, line)) {
            std::vector<std::string_view> columns = absl::StrSplit(line, '|');

            std::string lat(columns[1]);
            std::string isPref(columns[6]);
            std::string aui(columns[7]);
            std::string sab(columns[11]);
            std::string code(columns[13]);
            std::string name(columns[14]);
            std::string suppress(columns[16]);

            // filter out entries with noisy UMLS attributes
            if (isPref != "Y" || lat != "ENG") {
                continue;
            }

            if (suppress == "O" || suppress == "Y") {
                continue;
            }

            // filter out entries with short str name (<= 3 characters)
            if (name.size() <= 3) {
                continue;
            }

            // filter out entries without valid aui in our code_to_aui_map
            auto find_aui = code_to_aui_map.find(std::make_pair(sab, code));
            if (find_aui == std::end(code_to_aui_map)) {
                continue;
            }

            auto [iter, added] =
                result.insert(std::make_pair(find_aui->second, name));
            if (!added) {
                std::string prev_name = iter->second;
                if (prev_name.size() <= name.size()) {
                    continue;
                } else {
                    iter->second = name;
                }
            }
        }

        // add placeholder "NO_DEF" for aui with no definition
        for (auto& iter : code_to_aui_map) {
            auto find_aui = result.find(iter.second);
            if (find_aui == result.end()) {
                result.insert(std::make_pair(iter.second, "NO_DEF"));
            }
        }

        return result;
    }
};

#endif
