#include <nlohmann/json.hpp>

#include "absl/container/flat_hash_map.h"
#include "database.hh"
using json = nlohmann::json;

boost::filesystem::path extract =
<<<<<<< HEAD
    "/local-scratch/nigam/projects/ethanid/femr_9_extract";
=======
    "/local-scratch/nigam/projects/ethanid/"
<<<<<<< HEAD
    "som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v3";
>>>>>>> c79a258... Next
=======
    "som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2";
>>>>>>> 30c6f85... Next

int main() {
    PatientDatabase database(extract, true);

    const Dictionary& dict = database.get_code_dictionary();

    absl::flat_hash_map<uint32_t, uint32_t> length_counts;

    auto iter = database.iterator();

    std::string_view target = "STANFORD_OBS";

    uint32_t total = 0;

    for (uint32_t patient_id = 0; patient_id < database.size(); patient_id++) {
        const Patient& p = iter.get_patient(patient_id);

        uint32_t valid_events = 0;

        bool has_ip = false;

        for (const auto& event : p.events) {
            std::string_view code_str = dict[event.code];
            if (code_str.substr(0, target.size()) != target) {
                valid_events += 1;
            }
            if (event.code == 580) {
                has_ip = true;
            }
            length_counts[(uint32_t)event.value_type] += 1;
        }
    }

<<<<<<< HEAD
    for (uint32_t i = 0; i < 4; i++) {
        std::cout << i << " " << length_counts[i] << std::endl;
=======
        if (false && !has_ip) {
            continue;
        }

        if (patient_id == 0) {
            std::cout << valid_events << std::endl;
        }

        // length_counts[valid_events] += 1;
        length_counts[valid_events] += 1;
	total += valid_events;
>>>>>>> 30c6f85... Next
    }

    std::cout<<"Got " << total << std::endl;

    // std::ofstream o(
    //     "/local-scratch/nigam/projects/ethanid/femr/native/results/"
    //     "final_counts");
    // o << json(length_counts);
}
