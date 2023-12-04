#include <nlohmann/json.hpp>

#include "absl/container/flat_hash_map.h"
#include "database.hh"
using json = nlohmann::json;

boost::filesystem::path extract =
    "/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8";
//    "/share/pi/nigam/ethanid/debug_extract_issues/extract";

int main() {
    PatientDatabase database(extract, true);

    const Dictionary& dict = database.get_code_dictionary();

    absl::flat_hash_map<std::string, uint32_t> code_counts;

    std::cout<<dict.size()<<std::endl;

    for (uint32_t i = 0; i < dict.size(); i++) {
        std::string code = std::string(dict[i]);
        if (code_counts[code] > 0) {
            std::cout<<"WHat " << i << " " << code << " " << (code_counts[code] -1) << std::endl;
        }
        code_counts[code] = i + 1;
    }
}
