#include "survival_metrics.hh"

#include <boost/container/vector.hpp>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

TEST(SurvivalTest, SimpleSurvivalTest) {
    std::vector<double> times = {24.190982539140663,
                                 9.213163096445884,
                                 30.861266320008063,
                                 29.436412468622535,
                                 3,
                                 12.526487529295311,
                                 14.363129907923415,
                                 8.700571069475394,
                                 3,
                                 5.0085967727121785};
    boost::container::vector<bool> is_censor = {0, 0, 1, 1, 0, 1, 1, 1, 0, 0};
    std::vector<double> time_bins = {20, 30};

    Eigen::Tensor<double, 2> hazards(10, 2);

    std::vector<double> start = {0.7732234055519118, 0.3884639751732126,
                                 0.5410659924743124, 0.40147299931921016,
                                 0.341970365045018,  0.5259581728990739,
                                 0.5248952487113505, 0.5149263476688059,
                                 0.0862297362953266, 0.19539490942552173};
    std::vector<double> end = {0.42342818523948805,   0.13079267256871957,
                               0.49563579607725033,   0.21183335726151165,
                               0.17390282523638617,   0.28445105359172823,
                               0.41289203135105185,   0.429995700460779,
                               0.0032874323990542687, 0.08806888969182763};

    for (int i = 0; i < 10; i++) {
        hazards(i, 0) = start[i];
        hazards(i, 1) = end[i];
    }

    std::cout << "starting" << std::endl;

    double result = compute_c_statistic(times, is_censor, time_bins, hazards);

    std::cout << "Got " << result << std::endl;
}

TEST(SurvivalTest, CalibrationTest) {
    std::vector<double> probs = {0.01, 0.1, 0.2, 0.3, 0.4,  0.5,
                                 0.6,  0.7, 0.8, 0.9, 0.99, 0.5};
    boost::container::vector<bool> is_censor = {0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 1};

    std::vector<double> results = compute_calibration(probs, is_censor, 10);

    for (const auto& res : results) {
        std::cout << "Got " << res << std::endl;
    }
}