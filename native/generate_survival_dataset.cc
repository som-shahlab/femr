#include <map>

#include "database.hh"
#include "flatmap.hh"

struct SurvivalDatasetResult {
    std::map <
};

void combine_time_data(SurvivalDatasetResult& target,
                       const SurvivalDatasetResult& source) {}

void process_time_patient(SurvivalDatasetResult& data, const Patient& p) {}

int main() {
    std::map<std::string, std::string> targets = {
        {"pancreatic_cancer", "SNOMED/372003004"},
        {"celiac_disease", "SNOMED/396331005"},
        {"lupus", "SNOMED/55464009"},
        {"heart_attack", "SNOMED/57054005"},
        {"stroke", "SNOMED/432504007"},
        {"NAFL", "SNOMED/197321007"},
    };

    boost::filesystem::path path =
        "/local-scratch/nigam/projects/ethanid/"
        "som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2";

    PatientDatabase database(path, true);
    Ontology& ontology = database.get_ontology();

    std::map<uint32_t, std::string> target_codes;

    for (const auto& entry : targets) {
        auto code = ontology.get_dictionary().find(entry.second);
        if (!code) {
            throw std::runtime_error("Could not find " + entry.second);
        }
        std::cout << "Mapping " << *code << " to " << entry.first << std::endl;
        target_codes[*code].push_back(entry.first);
    }

    FlatMap<std::vector<std::string>> targets_per_code;

    for (uint32_t code = 0; code < ontology.get_dictionary().size(); code++) {
        for (uint32_t parent : ontology.get_all_parents(code)) {
            auto iter = target_codes.find(parent);
            if (iter != std::end(target_codes)) {
                targets_per_code
                    .find_or_insert(code, std::vector<std::string>())
                    ->push_back(iter->second);
            }
        }
    }
}
