#include "patient_collection_extension.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "absl/time/civil_time.h"
#include "civil_day_caster.h"
#include "reader.h"
#include "parse_utils.h"
#include "csv.h"
#include "register_iterable.h"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>

namespace py = pybind11;

enum class ValueType {
    NONE,
    NUMERIC,
    TEXT,
};

struct Value {
    ValueType type;
    std::string text_value;
    float numeric_value;
};

struct Event {
    absl::CivilDay start;
    absl::CivilDay end;

    float start_age;
    std::string code;
    Value value;

    bool has_end;
    
    float end_age;
    std::string event_type;
    int id;
    int parent_id;
};

struct Patient {
    uint32_t patient_id;
    absl::CivilDay birth_date;
    std::vector<Event> events;

    absl::Span<const Event> get_events() const { return events; }
};

/** Currently using a dummy reader designed to be as simple as possible for easy changes **/
class TimelineReader {
   public:
    TimelineReader(const char* filename) {
        for (auto&& file :
         boost::filesystem::recursive_directory_iterator(filename)) {
        auto path = file.path();

        if (boost::filesystem::is_directory(path)) {
            continue;
        }


        std::vector<std::string_view> columns = {
            "patient_id", "start", "end", "code", "value", "event_type", "id", "parent_id"
        };


        Patient patient;
        bool is_first = true;
        bool found_birth;

        auto add_patient = [&]() {
            if (found_birth) {
                for (auto& event : patient.events) {
                    event.start_age = event.start - patient.birth_date;
                    if (event.has_end) {
                        event.end_age = event.end - patient.birth_date;
                    } else {
                        event.end_age = -1;
                    }
                }

                patients[patient.patient_id] = patient;
            } else {
                std::cout<<"Warning, could not find birth for " << patient.patient_id << std::endl;
            }
        };

        csv_iterator(path.c_str(), columns, ',', {}, [&](const std::vector<std::string_view> &row) {
            uint32_t patient_id;
            attempt_parse_or_die(row[0], patient_id);

            if (is_first) {
                patient.patient_id = patient_id;
                is_first = false;
                found_birth = false;
            }

            if (patient_id != patient.patient_id) {
                add_patient();
                patient = Patient();
                patient.patient_id = patient_id;
                found_birth = false;
            }

            absl::CivilDay start = parse_date(row[1]);

            if (row[3] == "birth") {
                patient.birth_date = start;
                found_birth = true;
            } else {
                // Need to add to events
                Event event = {};
                event.start = start;

                event.has_end = row[2] != "";

                if (row[2] != "") {
                    event.end = parse_date(row[2]);
                }
                event.code = row[3];
                
                Value v;
                if (row[4] == "") {
                    v.type = ValueType::NONE;
                } else {
                    v.type = ValueType::TEXT;
                    v.text_value = row[4];
                }

                event.value = v;

                event.event_type = row[5];
                if (row[6] != "") {
            attempt_parse_or_die(row[6], event.id);
                }
                if (row[7] != "") {
            attempt_parse_or_die(row[7], event.parent_id);
                }
                patient.events.push_back(event);
            }
        });
        if (!is_first) {
            add_patient();
        }
    }

    }

    absl::Span<const uint32_t> get_patient_ids() const {
        return patient_ids;
    }

    Patient get_patient(uint32_t patient_id) {
        return patients[patient_id];
    }

    void close() {
        // add file close ops when necessary
    }

   private:
    std::map<uint32_t, Patient> patients;
    std::vector<uint32_t> patient_ids;
};

void convert_patients_to_extract(std::string patient_directory, std::string extract_file, int num_threads) {
    /* 
        Currently we do a nullopt extractor and use the raw gzip files as format. Could obviously be optimized as necessary.
    */
   namespace fs = boost::filesystem;

   boost::filesystem::path sourceDir(patient_directory);
   boost::filesystem::path destinationDir(extract_file);

    if (!fs::exists(sourceDir) || !fs::is_directory(sourceDir))
    {
        throw std::runtime_error("Source directory " + sourceDir.string() + " does not exist or is not a directory");
    }
    if (fs::exists(destinationDir))
    {
        throw std::runtime_error("Destination directory " + destinationDir.string() + " already exists");
    }
    if (!fs::create_directory(destinationDir))
    {
        throw std::runtime_error("Cannot create destination directory " + destinationDir.string());
    }

    for (const auto& dirEnt : fs::recursive_directory_iterator{sourceDir})
    {
        const auto& path = dirEnt.path();
        auto relativePathStr = path.string();
        boost::algorithm::replace_first(relativePathStr, sourceDir.string(), "");
        fs::copy(path, destinationDir / relativePathStr);
    }
}

void register_patient_collection_extension(py::module& root) {
    register_iterable<absl::Span<const Event>>(root);
    register_iterable<absl::Span<const uint32_t>>(root);

    py::module m = root.def_submodule("patient_collection");

    m.def("convert_patients_to_patient_collection", convert_patients_to_extract);

    py::class_<TimelineReader>(m, "PatientCollectionReader")
        .def(py::init<const char*>(), py::arg("filename"))
        .def("get_patient", &TimelineReader::get_patient)
        .def("get_patient_ids", &TimelineReader::get_patient_ids,
             py::keep_alive<0, 1>())
        .def("close", &TimelineReader::close);

    py::enum_<ValueType>(m, "ValueType")
        .value("NONE", ValueType::NONE)
        .value("TEXT", ValueType::TEXT)
        .value("NUMERIC", ValueType::NUMERIC);

    py::class_<Value>(m, "Value")
        .def_readonly("type", &Value::type)
        .def_readonly("text_value", &Value::text_value)
        .def_readonly("numeric_value", &Value::numeric_value);
    
    py::class_<Event>(m, "Event")
        .def_readonly("start_age", &Event::start_age)
        .def_readonly("code", &Event::code)
        .def_readonly("value", &Event::value)
        .def_readonly("end_age", &Event::end_age)
        .def_readonly("event_type", &Event::event_type)
        .def_readonly("id", &Event::id)
        .def_readonly("parent_id", &Event::parent_id);
    
    py::class_<Patient>(m, "Patient")
        .def_readonly("patient_id", &Patient::patient_id)
        .def_readonly("birth_date", &Patient::birth_date)
        .def_property_readonly("events", &Patient::get_events,
                               py::keep_alive<0, 1>());
}
