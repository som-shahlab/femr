#include <nlohmann/json.hpp>

#include "absl/container/flat_hash_map.h"
#include "database.hh"
using json = nlohmann::json;

boost::filesystem::path extract =
    "/local-scratch/nigam/projects/ethanid/femr_9_extract";

int main() {
    PatientDatabase database(extract, true);

    const Dictionary& dict = database.get_code_dictionary();

    absl::flat_hash_map<uint32_t, uint32_t> length_counts;

    auto iter = database.iterator();

    std::string_view target = "STANFORD_OBS";

    uint32_t total = 0;

    for (uint32_t patient_offset = 0; patient_offset < database.size();
         patient_offset++) {
        const Patient& p = iter.get_patient(patient_offset);

        uint32_t valid_events = 0;

        bool has_ip = false;

        for (const auto& event : p.events) {
            std::string_view code_str = dict[event.code];
            if (code_str.substr(0, target.size()) != target) {
                valid_events += 1;
            }
            length_counts[(uint32_t)event.value_type] += 1;
        }
    }

    std::cout << "Got " << total << std::endl;

    // std::ofstream o(
    //     "/local-scratch/nigam/projects/ethanid/femr/native/results/"
    //     "final_counts");
    // o << json(length_counts);
}
