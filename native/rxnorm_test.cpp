#include "rxnorm.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

class RxNormTest : public ::testing::Test {
   public:
    RxNorm rxnorm;
};

TEST_F(RxNormTest, ConvertToATC) {
    ASSERT_THAT(
        rxnorm.get_atc_codes("NDC", "00009028052"),
        ::testing::ElementsAre("ATC/D07AA01", "ATC/D10AA02", "ATC/H02AB04"));

    ASSERT_THAT(
        rxnorm.get_atc_codes("HCPCS", "J1644"),
        ::testing::ElementsAre("ATC/B01AB01", "ATC/C05BA03", "ATC/S01XA14"));
}