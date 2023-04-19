const char* extract_location = "/local-scratch/nigam/projects/ethanid/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8";
const char* source_dict_location = "/local-scratch/nigam/projects/clmbr_text_assets/data/clmbr_lr_0.0001_wd_0.0_id_0.0_td_0.0_rt_global_maxiter_10000000_hs_768_is_3072_nh_12_nl_6_aw_512_obs/dictionary";
const char* destination_dict_location = "/local-scratch/nigam/projects/ethanid/femr_develop/femr/native/output_dict";

#include "database.hh"


#include <nlohmann/json.hpp>

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
        entry["text_string"] = data.get_shared_text_dictionary()[entry["text_value"]];
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
    j["regular"] = convert_entries(database, input["regular"]);
    j["ontology_rollup"] = convert_entries(database, input["ontology_rollup"]);
    j["age_stats"] = input["age_stats"];
    j["hierarchical_counts"] = input["hierarchical_counts"];

    std::vector<std::uint8_t> v = json::to_msgpack(j);

    std::ofstream o(destination_dict_location, std::ios_base::binary);

    o.write((const char*)v.data(), v.size());
}
