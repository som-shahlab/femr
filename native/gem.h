#ifndef GEM_H_INCLUDED
#define GEM_H_INCLUDED

#include <fstream>
#include <sstream>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/substitute.h"

class GEMMapper {
   public:
    GEMMapper(std::string_view gem_location) {
        helper(gem_location, "2018_I9gem.txt", diag_convert_map, true);
        helper(gem_location, "gem_i9pcs.txt", proc_convert_map, false);
    }

    void helper(
        std::string_view gem_location, std::string_view path,
        absl::flat_hash_map<std::string, std::vector<std::string>>& target,
        bool diag) {
        std::string full_filename =
            absl::Substitute("$0/$1", gem_location, path);
        std::ifstream infile(full_filename);

        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            std::string icd9, icd10, flags;
            iss >> icd9 >> icd10 >> flags;

            if (diag) {
                if (icd10.size() > 3) {
                    icd10.insert(3, ".");
                }
            }

            bool no_map = flags[1] == '1';
            bool combination = flags[2] == '1';
            int scenario = flags[3] - '0';
            int choice_list = flags[4] - '0';

            if (no_map) {
                continue;
            }

            if (!combination) {
                if (target[icd9].size() == 0) {
                    target[icd9] = {icd10};
                }
            } else {
                if (scenario != 1) {
                    continue;
                }

                if (target[icd9].size() < (size_t)std::max(2, choice_list)) {
                    target[icd9].resize(std::max(2, choice_list));
                }
                target[icd9][choice_list - 1] = icd10;
            }
        }
    }

    std::string normalize(std::string_view piece) const {
        std::string copy(piece);
        copy.erase(std::remove(copy.begin(), copy.end(), '.'), copy.end());
        return copy;
    }

    std::vector<std::string> map_diag(std::string_view code) const {
        auto iter = diag_convert_map.find(normalize(code));

        if (iter == std::end(diag_convert_map)) {
            return {};
        } else {
            return iter->second;
        }
    }

    std::vector<std::string> map_proc(std::string_view code) const {
        auto iter = proc_convert_map.find(normalize(code));

        if (iter == std::end(proc_convert_map)) {
            return {};
        } else {
            return iter->second;
        }
    }

   private:
    absl::flat_hash_map<std::string, std::vector<std::string>> proc_convert_map;
    absl::flat_hash_map<std::string, std::vector<std::string>> diag_convert_map;
};

#endif