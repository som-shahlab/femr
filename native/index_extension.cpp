#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "reader.h"

namespace py = pybind11;

void create_index(std::string parent_timelines, std::string output_filename) {
    ExtractReader extract(parent_timelines.c_str(), true);
    ExtractReaderIterator iterator = extract.iter();

    absl::flat_hash_map<uint32_t, std::vector<uint32_t>> patients_per_code;

    std::cout << "Starting to process" << std::endl;

    std::vector<uint32_t> codes;
    uint32_t processed = 0;
    for (uint32_t patient_id : extract.get_patient_ids()) {
        processed += 1;

        if (processed % 1000000 == 0) {
            std::cout << absl::Substitute("Processed $0 out of $1", processed,
                                          extract.get_patient_ids().size())
                      << std::endl;
        }
        codes.clear();
        bool found = iterator.process_patient(
            patient_id, [&codes](absl::CivilDay birth_date, uint32_t age,
                                 const std::vector<uint32_t>& observations,
                                 const std::vector<ObservationWithValue>&
                                     observations_with_values) {
                for (uint32_t obs : observations) {
                    codes.push_back(obs);
                }

                for (auto obs_with_value : observations_with_values) {
                    codes.push_back(obs_with_value.code);
                }
            });

        if (!found) {
            std::cout << absl::Substitute("Could not find patient id $0",
                                          patient_id)
                      << std::endl;
            abort();
        }

        std::sort(std::begin(codes), std::end(codes));
        codes.erase(std::unique(std::begin(codes), std::end(codes)),
                    std::end(codes));

        for (uint32_t code : codes) {
            patients_per_code[code].push_back(patient_id);
        }
    }

    ConstdbWriter writer(output_filename.c_str());

    std::vector<uint8_t> compressed_buffer;
    for (auto& item : patients_per_code) {
        uint32_t code_id = item.first;
        std::vector<uint32_t>& patient_ids = item.second;

        std::sort(std::begin(patient_ids), std::end(patient_ids));

        uint32_t last_id = 0;

        for (uint32_t& pid : patient_ids) {
            uint32_t delta = pid - last_id - 1;
            last_id = pid;
            pid = delta;
        }

        size_t max_needed_size =
            streamvbyte_max_compressedbytes(patient_ids.size()) +
            sizeof(uint32_t);

        if (compressed_buffer.size() < max_needed_size) {
            compressed_buffer.resize(max_needed_size * 2 + 1);
        }

        size_t actual_size =
            streamvbyte_encode(patient_ids.data(), patient_ids.size(),
                               compressed_buffer.data() + sizeof(uint32_t));

        uint32_t* start_of_compressed_buffer =
            reinterpret_cast<uint32_t*>(compressed_buffer.data());
        *start_of_compressed_buffer = patient_ids.size();

        writer.add_int(code_id, (const char*)compressed_buffer.data(),
                       actual_size + sizeof(uint32_t));
    }
}

void register_index_extension(py::module& root) {
    py::module m = root.def_submodule("index");
    py::class_<Index>(m, "Index")
        .def(py::init<const char*>(), py::arg("filename"))
        .def("get_patient_ids", &Index::get_patient_ids, py::arg("term"))
        .def("get_all_patient_ids", &Index::get_all_patient_ids,
             py::arg("terms"));
    
    m.def("create_index", create_index);
}