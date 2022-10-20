#include <nlohmann/json.hpp>

#include "absl/container/flat_hash_map.h"
#include "database.hh"
using json = nlohmann::json;

boost::filesystem::path extract =
    "/local-scratch/nigam/projects/ethanid/piton/targetb/"
    "omop_extractor_40flu58k/targetb";

int main() {
    PatientDatabase database(extract, true);

    absl::flat_hash_map<uint32_t, uint32_t> length_counts;

    auto iter = database.iterator();

    for (uint32_t patient_id = 0; patient_id < database.size(); patient_id++) {
        const Patient& p = iter.get_patient(patient_id);

        length_counts[p.events.size()] += 1;
    }

    std::ofstream o("counts");
    o << json(length_counts);
}
