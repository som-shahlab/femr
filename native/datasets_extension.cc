#include "datasets_extension.hh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "absl/strings/str_cat.h"
#include "absl/time/civil_time.h"
#include "civil_day_caster.hh"
#include "database.hh"
#include "database_test_helper.hh"
#include "filesystem_caster.hh"
#include "join_csvs.hh"
#include "register_iterable.hh"

namespace py = pybind11;

void keep_alive(py::handle nurse, py::handle patient) {
    /* Fall back to clever approach based on weak references taken from
     * Boost.Python. This is not used for pybind-registered types because
     * the objects can be destroyed out-of-order in a GC pass. */
    py::cpp_function disable_lifesupport([patient](py::handle weakref) {
        patient.dec_ref();
        weakref.dec_ref();
    });

    py::weakref wr(nurse, disable_lifesupport);

    patient.inc_ref(); /* reference patient and leak the weak reference */
    (void)wr.release();
}

namespace {

class OntologyWrapper {
   public:
    OntologyWrapper(Ontology& _ontology) : ontology(_ontology) {}

    py::str get_code_str(uint32_t code_index) {
        if (code_index >= main_dictionary.size()) {
            main_dictionary.resize((code_index + 1) * 2);
        }
        auto& val = main_dictionary[code_index];
        if (!val.has_value()) {
            val.emplace(ontology.get_dictionary()[code_index]);
        }
        return *val;
    }

    template <typename F>
    py::tuple get_generic(std::string_view code_str, F f) {
        auto possible_entry = ontology.get_dictionary().find(code_str);
        if (!possible_entry) {
            throw py::index_error();
        }
        auto result = f(*possible_entry);
        py::tuple converted_result(result.size());

        for (size_t i = 0; i < result.size(); i++) {
            converted_result[i] = get_code_str(result[i]);
        }

        return converted_result;
    }

    py::tuple get_parents(std::string_view code_str) {
        return get_generic(code_str, [this](uint32_t code) {
            return ontology.get_parents(code);
        });
    }

    py::tuple get_children(std::string_view code_str) {
        return get_generic(code_str, [this](uint32_t code) {
            return ontology.get_children(code);
        });
    }

    py::tuple get_all_parents(std::string_view code_str) {
        return get_generic(code_str, [this](uint32_t code) {
            return ontology.get_all_parents(code);
        });
    }

    std::string_view get_text_description(std::string_view code_str) {
        auto possible_entry = ontology.get_dictionary().find(code_str);
        if (!possible_entry) {
            throw py::index_error();
        }
        return ontology.get_text_description(*possible_entry);
    }

    py::str get_code_from_concept_id(uint64_t concept_id) {
        auto possible_entry = ontology.get_code_from_concept_id(concept_id);
        if (!possible_entry) {
            throw py::index_error();
        }
        return get_code_str(*possible_entry);
    }

    uint64_t get_concept_id_from_code(std::string_view code_str) {
        auto possible_entry = ontology.get_dictionary().find(code_str);
        if (!possible_entry) {
            throw py::index_error();
        }
        return ontology.get_concept_id_from_code(*possible_entry);
    }

   private:
    std::vector<boost::optional<py::str>> main_dictionary;
    Ontology& ontology;
};

class PatientDatabaseWrapper : public PatientDatabase {
   public:
    PatientDatabaseWrapper(const boost::filesystem::path& path, bool read_all,
                           bool read_all_unique_text = false)
        : PatientDatabase(path, read_all, read_all_unique_text),
          ontology_wrapper(get_ontology()) {}

    OntologyWrapper& get_ontology_wrapper() { return ontology_wrapper; }

   private:
    OntologyWrapper ontology_wrapper;
};

class EventWrapper {
   public:
    EventWrapper(py::module pickle, py::object python_event,
                 PatientDatabaseWrapper* database, uint32_t patient_offset,
                 absl::CivilSecond birth_date, uint32_t event_index,
                 const Event& event)
        : m_pickle(pickle),
          m_python_event(python_event),
          m_database(database),
          m_patient_offset(patient_offset),
          m_birth_date(birth_date),
          m_event_index(event_index),
          m_event(event) {}

    py::object code() {
        if (!m_code) {
            m_code.emplace(
                m_database->get_ontology_wrapper().get_code_str(m_event.code));
        }
        return *m_code;
    }

    py::object start() {
        if (!m_start) {
            absl::CivilSecond start_time =
                m_birth_date + 60 * m_event.start_age_in_minutes;

            m_start.emplace(py::cast(start_time));
        }
        return *m_start;
    }

    py::object value() {
        if (!m_value) {
            switch (m_event.value_type) {
                case ValueType::NONE:
                    m_value.emplace(py::none());
                    break;

                case ValueType::NUMERIC:
                    m_value.emplace(py::cast(m_event.numeric_value));
                    break;

                case ValueType::UNIQUE_TEXT:
                case ValueType::SHARED_TEXT: {
                    std::string_view data;

                    if (m_event.value_type == ValueType::UNIQUE_TEXT) {
                        auto dict = m_database->get_unique_text_dictionary();
                        if (dict == nullptr) {
                            data = "";
                        } else {
                            data = (*dict)[m_event.text_value];
                        }
                    } else {
                        data = m_database->get_shared_text_dictionary()
                                   [m_event.text_value];
                    }

                    m_value.emplace(py::str(data.data(), data.size()));
                    break;
                }

                default:
                    throw std::runtime_error("Invalid value?");
            }
        }

        return *m_value;
    }

    py::object metadata() {
        if (!m_metadata) {
            std::string_view metadata_str =
                m_database->get_event_metadata(m_patient_offset, m_event_index);
            if (metadata_str.size() > 0) {
                py::object bytes =
                    py::bytes(metadata_str.data(), metadata_str.size());
                m_metadata.emplace(m_pickle.attr("loads")(bytes));
            } else {
                m_metadata.emplace(py::dict());
            }
        }
        return *m_metadata;
    }

    py::object to_python_event() {
        using namespace pybind11::literals;
        return m_python_event("code"_a = code(), "start"_a = start(),
                              "value"_a = value(), **metadata());
    }

   private:
    py::module m_pickle;
    py::object m_python_event;

    PatientDatabaseWrapper* m_database;
    uint32_t m_patient_offset;
    absl::CivilSecond m_birth_date;
    uint32_t m_event_index;
    Event m_event;

    boost::optional<py::object> m_start;
    boost::optional<py::object> m_code;
    boost::optional<py::object> m_value;
    boost::optional<py::object> m_metadata;
};

}  // namespace

void register_datasets_extension(py::module& root) {
    py::module pickle = py::module::import("pickle");

    py::module abc = py::module::import("collections.abc");
    py::object abc_sequence = abc.attr("Sequence");
    py::object abc_mapping = abc.attr("Mapping");

    register_iterable<absl::Span<const uint32_t>>(root, "IntSpan");

    py::module m = root.def_submodule("datasets");
    py::object femr_root = py::module_::import("femr");
    if (!femr_root) {
        throw std::runtime_error("Could not import root");
    }
    py::object python_patient = femr_root.attr("Patient");
    if (!python_patient) {
        throw std::runtime_error("Could not import python patient");
    }
    py::object python_event = femr_root.attr("Event");
    if (!python_event) {
        throw std::runtime_error("Could not import python event");
    }

    {
        py::module test_module = m.def_submodule("test");
        test_module.def("create_ontology_files", create_ontology_files);
        test_module.def("create_database_files", create_database_files);
    }

    m.def("sort_and_join_csvs", [](std::string source_path,
                                   std::string target_path, py::object fields,
                                   char delimiter, int num_threads) {
        std::vector<std::pair<std::string, ColumnValueType>> column_types;

        if (py::isinstance<py::list>(fields)) {
            // Assume all string fields
            py::list list = py::reinterpret_borrow<py::list>(fields);
            for (auto item : list) {
                column_types.emplace_back(item.cast<std::string>(),
                                          ColumnValueType::STRING);
            }
        } else if (py::isinstance<py::dtype>(fields)) {
            py::dict fields_dict = fields.attr("fields");
            for (const auto& entry : fields_dict) {
                std::string name = entry.first.cast<std::string>();
                py::tuple type_and_offset = entry.second.cast<py::tuple>();
                py::dtype type = type_and_offset[0].cast<py::dtype>();
                ColumnValueType our_type;
                switch (type.kind()) {
                    case 'M':
                        our_type = ColumnValueType::DATETIME;
                        break;

                    case 'S':
                        our_type = ColumnValueType::STRING;
                        break;

                    case 'u':
                        our_type = ColumnValueType::UINT64_T;
                        break;
                    default:
                        throw std::runtime_error(absl::StrCat(
                            "Invalid kind ", std::to_string(type.kind())));
                }
                column_types.emplace_back(name, our_type);
            }
        } else {
            throw std::runtime_error(
                "Invalid type passed as fields to sort_and_join_csvs");
        }
        sort_and_join_csvs(source_path, target_path, column_types, delimiter,
                           num_threads);
    });

    m.def("convert_patient_collection_to_patient_database",
          convert_patient_collection_to_patient_database);

    py::class_<PatientDatabaseWrapper> database_binding(m, "PatientDatabase");

    database_binding
        .def(py::init<std::string, bool, bool>(), py::arg("filename"),
             py::arg("read_all") = false,
             py::arg("read_all_unique_text") = false)
        .def("__len__",
             [](PatientDatabaseWrapper& self) { return self.size(); })
        .def(
            "__getitem__",
            [python_patient, python_event, pickle](PatientDatabaseWrapper& self,
                                                   uint64_t patient_id) {
                using namespace pybind11::literals;

                boost::optional<uint32_t> patient_offset =
                    self.get_patient_offset(patient_id);

                if (!patient_offset) {
                    throw py::index_error();
                }
                Patient p = self.get_patient(*patient_offset);
                py::tuple events(p.events.size());

                absl::CivilSecond birth_date = p.birth_date;

                for (size_t i = 0; i < p.events.size(); i++) {
                    const Event& event = p.events[i];
                    events[i] =
                        EventWrapper(pickle, python_event, &self,
                                     *patient_offset, birth_date, i, event);
                }

                return python_patient("patient_id"_a = patient_id,
                                      "events"_a = events);
            },
            py::return_value_policy::reference_internal)
        .def(
            "__iter__",
            [](PatientDatabaseWrapper& self) {
                auto span = self.get_patient_ids();
                return py::make_iterator(std::begin(span), std::end(span));
            },
            py::return_value_policy::reference_internal)
        .def("get_patient_birth_date",
             [](PatientDatabaseWrapper& self, uint64_t patient_id) {
                 Patient p =
                     self.get_patient(*self.get_patient_offset(patient_id));
                 return p.birth_date;
             })
        .def("get_ontology", &PatientDatabaseWrapper::get_ontology_wrapper,
             py::return_value_policy::reference_internal)
        .def("compute_split",
             [](PatientDatabaseWrapper& self, uint32_t seed,
                uint64_t patient_id) {
	         auto potential_offset = self.get_patient_offset(patient_id);
		 if (!potential_offset) {
                    throw py::index_error();
		 }

                 return self.compute_split(seed, *potential_offset);
             })
        .def("version_id", &PatientDatabaseWrapper::version_id)
        .def("database_id", &PatientDatabaseWrapper::database_id)
        .def("close",
             [](const PatientDatabaseWrapper& self) {
                 // TODO: Implement this to save memory and file pointers
             })
        .attr("__bases__") =
        py::make_tuple(abc_mapping) + database_binding.attr("__bases__");

    py::class_<OntologyWrapper>(m, "Ontology")
        .def("get_parents", &OntologyWrapper::get_parents)
        .def("get_children", &OntologyWrapper::get_children)
        .def("get_all_parents", &OntologyWrapper::get_all_parents)
        .def("get_text_description", &OntologyWrapper::get_text_description)
        .def("get_code_from_concept_id",
             &OntologyWrapper::get_code_from_concept_id)
        .def("get_concept_id_from_code",
             &OntologyWrapper::get_concept_id_from_code);

    py::class_<EventWrapper>(m, "EventWrapper")
        .def_property_readonly("code", &EventWrapper::code)
        .def_property_readonly("start", &EventWrapper::start)
        .def_property_readonly("value", &EventWrapper::value)
        .def("__getattr__",
             [](EventWrapper& wrapper, const std::string& attr) {
                 return wrapper.metadata().attr("get")(attr, py::none());
             })
        .def("__repr__", [python_event](EventWrapper& wrapper) {
            return py::str(wrapper.to_python_event());
        });

    py::class_<Dictionary> dictionary_binding(m, "Dictionary");
    dictionary_binding
        .def("__len__", [](Dictionary& self) { return self.size(); })
        .def("__getitem__",
             [](Dictionary& self, uint32_t index) {
                 if (index >= self.size()) {
                     throw py::index_error();
                 }
                 auto data = self[index];
                 return py::str(data.data(), data.size());
             })
        .def("index",
             [](Dictionary& self, std::string data) {
                 auto iter = self.find(data);
                 if (!iter) {
                     throw py::value_error();
                 }
                 return *iter;
             })
        .attr("__bases__") =
        py::make_tuple(abc_sequence) + dictionary_binding.attr("__bases__");
}
