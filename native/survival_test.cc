#include "survival.hh"

#include "gmock/gmock.h"
#include "gtest/gtest.h"


TEST(SurvivalTest, TestSurvival) {
    Patient example_patient;
    example_patient.birth_date = absl::CivilDay(1990, 1, 1);
    example_patient.events.resize(3);
    example_patient.events[0].start_age_in_minutes = 200;
    example_patient.events[0].code = 10;
    example_patient.events[1].start_age_in_minutes = 400;
    example_patient.events[1].code = 20;
    example_patient.events[2].start_age_in_minutes = 600;
    example_patient.events[2].code = 30;

    SurvivalCalculator calculator;
    calculator.preprocess_patient(example_patient, [](uint32_t code) {std::vector<uint32_t> result; result.push_back(code); return result;});

    auto res = calculator.get_times_for_event(0);
    EXPECT_EQ(res.first, 600);
    std::vector<std::pair<uint32_t, uint32_t>> expected = {{200, 10}, {400, 20}, {600, 30}};
    EXPECT_EQ(res.second, expected);

    res = calculator.get_times_for_event(200);
    EXPECT_EQ(res.first, 400);
    expected = {{200, 20}, {400, 30}};
    EXPECT_EQ(res.second, expected);

    calculator.preprocess_patient(example_patient, [](uint32_t code) {std::vector<uint32_t> result; result.push_back(code); return result;}, 500);

    res = calculator.get_times_for_event(0);
    EXPECT_EQ(res.first, 500);
    expected = {{200, 10}, {400, 20}};
    EXPECT_EQ(res.second, expected);

    res = calculator.get_times_for_event(200);
    EXPECT_EQ(res.first, 300);
    expected = {{200, 20}};
    EXPECT_EQ(res.second, expected);
}
