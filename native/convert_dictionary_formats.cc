const char* extract_location = "/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8";

const char* source_dict_location = "/local-scratch/nigam/projects/ethanid/text_checks/clmbr_dictionary_fixed";
const char* destination_dict_location = "/local-scratch/nigam/projects/ethanid/text_checks/clmbr_dictionary_fixed_with_ontology";

#include <nlohmann/json.hpp>

#include "database.hh"

using json = nlohmann::json;
#include <fstream>

json read_file(std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    return json::from_msgpack(file);
}

json convert_entries(PatientDatabase& data, json entries) {
    std::vector<json> result;

    const json::array_t* actual_data = entries.get_ptr<const json::array_t*>();

    for (uint32_t i = 0; i < actual_data->size(); i++) {
        json entry = (*actual_data)[i];
        entry["code_string"] = data.get_code_dictionary()[entry["code"]];
        entry["text_string"] =  "";
	if (entry["type"] == "2") {
		entry["type"] = "3";
	}
        entry.erase("text_value");
        entry.erase("code");
        result.push_back(entry);
    }

    return result;
}

int main() {
    PatientDatabase database(extract_location, false, false);

    json input = read_file(source_dict_location);

    json j;
    j["regular"] = input["regular"];
    j["ontology_rollup"] = input["ontology_rollup"];
    j["age_stats"] = input["age_stats"];
    j["hierarchical_counts"] = input["hierarchical_counts"];

    std::map<std::string, std::vector<std::string>> all_parents_map;

    for (uint32_t i = 0; i < database.get_ontology().get_dictionary().size(); i++) {
        std::string entry = std::string(database.get_ontology().get_dictionary()[i]);

        std::vector<std::string> parents;
        for (uint32_t parent : database.get_ontology().get_all_parents(i)) {
            parents.push_back(std::string(database.get_ontology().get_dictionary()[parent]));
        }
        all_parents_map[entry] = parents;
    }

    j["all_parents"] = all_parents_map;

    std::vector<std::uint8_t> v = json::to_msgpack(j);

    std::ofstream o(destination_dict_location, std::ios_base::binary);

    o.write((const char*)v.data(), v.size());
}
