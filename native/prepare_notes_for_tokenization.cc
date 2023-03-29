#include <boost/algorithm/string/replace.hpp>
#include <boost/regex.hpp>
#include <fstream>

#include "absl/container/flat_hash_map.h"
#include "database.hh"
#include "parse_utils.hh"

boost::filesystem::path extract =
    "/local-scratch/nigam/projects/ethanid/"
    "som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract_v3";

int main() {
    PatientDatabase database(extract, true, false);

    std::cout << "Working with database " << database.size() << std::endl;
    std::cout << "Working with database "
              << database.get_shared_text_dictionary().size() << std::endl;

    std::cout << "Working with database "
              << database.get_unique_text_dictionary()->size() << std::endl;

    auto iter = database.iterator();

    std::ofstream output(
        "/local-scratch/nigam/projects/ethanid/femr/native/out_text");
    std::ofstream output_raw(
        "/local-scratch/nigam/projects/ethanid/femr/native/out_text_raw");

    boost::regex date_regex("\\d{2}/\\d{2}/\\d{4}");
    boost::regex longdash_regex("--+");
    boost::regex longunder_regex("__+");
    boost::regex longem_regex("——+");

    for (uint32_t patient_id = 0; patient_id < 10000; patient_id++) {
        const Patient& p = iter.get_patient(patient_id);

        // std::cout << "What " << patient_id << std::endl;
        absl::CivilMinute birth_minute = p.birth_date;

        for (const auto& event : p.events) {
            absl::CivilMinute current_day =
                event.start_age_in_minutes + birth_minute;

            absl::CivilDay current_date(current_day.year(), current_day.month(),
                                        current_day.day());

            if (false && event.value_type == ValueType::SHARED_TEXT) {
                if (database.get_shared_text_count(event.text_value) > 100) {
                    continue;
                }
                auto text =
                    database.get_shared_text_dictionary()[event.text_value];
                if (text.size() < 100) {
                    continue;
                }
                std::cout << "Event " << event.start_age_in_minutes << " "
                          << event.code << " " << (int)event.value_type
                          << std::endl;
                std::cout << "Got it " << text << " "
                          << database.get_shared_text_count(event.text_value)
                          << std::endl;
            }

            if (event.value_type == ValueType::UNIQUE_TEXT) {
                std::string text = std::string(
                    (*database.get_unique_text_dictionary())[event.text_value]);

                if (text.size() < 200) {
                    if (false) {
                        std::cout << "Lost "
                                  << database.get_code_dictionary()[event.code]
                                  << " " << text << std::endl;
                    }
                    continue;
                }

                std::string_view banned_prefix = "STANFORD_OBS";

                std::string_view text_str =
                    database.get_code_dictionary()[event.code];
                if (false &&
                    text_str.substr(0, banned_prefix.size()) == banned_prefix) {
                    std::cout << "Got banned " << text_str << " | " << text
                              << std::endl;
                }

                boost::replace_all(text, "\n", "<newline>");
                std::string fixed_text = text;

                // Hack to deal with bad unicode support ...
                boost::replace_all(fixed_text, "—", "-");

                fixed_text = boost::regex_replace(
                    fixed_text, date_regex,
                    [&](const boost::match_results<std::string::const_iterator>&
                            match) {
                        int day, month, year;
                        attempt_parse_or_die(match.str().substr(0, 2), month);
                        attempt_parse_or_die(match.str().substr(3, 2), day);
                        attempt_parse_or_die(match.str().substr(6, 4), year);
                        absl::CivilDay date(year, month, day);

                        // std::cout << "Found date " << match.str() << " " <<
                        // date
                        //           << " " << current_date << std::endl;

                        int delta = current_date - date;

                        return absl::StrCat(delta, " days");
                    });

                fixed_text =
                    boost::regex_replace(fixed_text, longdash_regex, "__");

                fixed_text =
                    boost::regex_replace(fixed_text, longunder_regex, "--");

                if (fixed_text.size() < 100) {
                    std::cout << "WAT? " << text << std::endl;
                    std::cout << "WAT? " << fixed_text << std::endl;
                }

                if (false) {
                    bool valid = false;
                    for (char c : fixed_text) {
                        if (c != ' ') {
                            valid = true;
                        }
                    }

                    if (!valid) {
                        std::cout
                            << "What " << patient_id << " "
                            << database.get_code_dictionary()[event.code] << " "
                            << database.get_original_patient_id(patient_id)
                            << " " << text.size() << " " << current_day << text
                            << " , " << fixed_text << std::endl;
                    }
                }

                if (false) {
                    std::cout << "Got " << text << std::endl;
                    std::cout << "Replaced with " << fixed_text << std::endl;
                    // exit(-1);
                }
                output << fixed_text << std::endl;
                output_raw << text << std::endl;
            }
        }
    }
}
