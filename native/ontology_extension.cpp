#include "ontology_extension.h"
#include "timeline_extension.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "constdb.h"
#include "reader.h"

namespace py = pybind11;

const std::vector<uint32_t> get_subwords(uint32_t code,
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>>& code_to_parents_map,
    const absl::flat_hash_map<uint32_t, uint32_t>& parent_map) {
    auto iter = code_to_parents_map.find(code);

    if (iter != std::end(code_to_parents_map)) {
        return iter->second;
    } else {
        std::vector<uint32_t> results;
        results.push_back(code);
        auto parent = parent_map.find(code);
        if (parent != std::end(parent_map)) {
            for (const auto& parent_code : get_subwords(parent->second, code_to_parents_map, parent_map)) {
                results.push_back(parent_code);
            }
        }
        code_to_parents_map[code] = results;

        return results;
    }
}

void create_ontology(const std::string& extract_path, const std::unordered_map<std::string, std::string>& parent_map, 
     const std::vector<std::string>& recorded_date_codes, const std::string& output_filename) {

    ExtractReader reader(extract_path.c_str(), false);
    const TermDictionary& dictionary = reader.get_dictionary();

    auto entries = dictionary.decompose();

    TermDictionary ontology_dictionary;
    OntologyCodeDictionary aui_text_description_dictionary;

    ConstdbWriter ontology_writer(output_filename.c_str());

    absl::flat_hash_map<uint32_t, std::vector<uint32_t>> code_to_parents_map;
    absl::flat_hash_map<uint32_t, uint32_t> converted_parent_map;
    for (const auto& entry : parent_map) {
        uint32_t child = ontology_dictionary.map_or_add(entry.first);
        uint32_t parent = ontology_dictionary.map_or_add(entry.second);

        converted_parent_map[child] = parent;
    }

    std::vector<uint32_t> converted_recorded_date_codes;

    for (const auto& c : recorded_date_codes) {
        auto value = ontology_dictionary.map(c);
        if (!value) {
            std::cout<<"Could not map " << c << std::endl;
            abort();
        }
        converted_recorded_date_codes.push_back(*value);
    }

    for (uint32_t i = 0; i < entries.size(); i++) {
        auto entry = entries[i];
        std::vector<uint32_t> subwords;
        auto val = ontology_dictionary.map(entry.first);
        if (val) {
            uint32_t code = *val;
            subwords = get_subwords(code, code_to_parents_map, converted_parent_map);
        }

        ontology_writer.add_int(i, (const char*)subwords.data(),
                                subwords.size() * sizeof(uint32_t));
    }

    for (auto& iter : code_to_parents_map) {
        auto& parent_codes = iter.second;
        std::sort(std::begin(parent_codes), std::end(parent_codes));

        int32_t subword_as_int = iter.first + 1;
        ontology_writer.add_int(-subword_as_int,
                                (const char*)parent_codes.data(),
                                parent_codes.size() * sizeof(uint32_t));
    }

    std::string dictionary_str = ontology_dictionary.to_json();
    ontology_writer.add_str("dictionary", dictionary_str.data(),
                            dictionary_str.size());
    std::string description_dictionary_str =
        aui_text_description_dictionary.to_json();
    ontology_writer.add_str("recorded_date_codes",
                            (const char*)recorded_date_codes.data(),
                            converted_recorded_date_codes.size() * sizeof(uint32_t));
    uint32_t root_node = *ontology_dictionary.map("SRC/V-SRC");
    ontology_writer.add_str("root", (const char*)&root_node, sizeof(uint32_t));
     

}

void register_ontology_extension(py::module& root) {
    py::module m = root.def_submodule("ontology");
    py::class_<OntologyReader>(m, "OntologyReader")
        .def(py::init<const char*>(), py::arg("filename") = std::nullopt)
        .def("get_subwords", &OntologyReader::get_subwords, py::arg("code"),
             py::keep_alive<0, 1>())
        .def("get_parents", &OntologyReader::get_parents, py::arg("subword"),
             py::keep_alive<0, 1>())
        .def("get_all_parents", &OntologyReader::get_all_parents,
             py::arg("subword"), py::keep_alive<0, 1>())
        .def("get_children", &OntologyReader::get_children, py::arg("subword"),
             py::keep_alive<0, 1>())
        .def("get_words_for_subword", &OntologyReader::get_words_for_subword,
             py::arg("code"), py::keep_alive<0, 1>())
        .def("get_words_for_subword_term",
             &OntologyReader::get_words_for_subword_term, py::arg("term"),
             py::keep_alive<0, 1>())
        .def("get_recorded_date_codes",
             &OntologyReader::get_recorded_date_codes, py::keep_alive<0, 1>())
        .def("get_dictionary", &OntologyReader::get_dictionary,
             py::return_value_policy::reference_internal)
        .def("get_text_description_dictionary",
             &OntologyReader::get_text_description_dictionary,
             py::return_value_policy::reference_internal);
    py::class_<OntologyCodeDictionary>(m, "OntologyCodeDictionary")
        .def("get_word", &OntologyCodeDictionary::get_word, py::arg("code"))
        .def("get_definition", &OntologyCodeDictionary::get_definition,
             py::arg("code"))
        .def("map", &OntologyCodeDictionary::map, py::arg("code"));
}
