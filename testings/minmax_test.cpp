#include <gtest/gtest.h>

#include <stdlib.h>
#include <limits>
#include <functional>
#include <algorithm>
#include <utility>
#include <opencv2/core/core.hpp>

#include <cvutil/algorithm/minmax.hpp>

TEST(algorithmTest, minmax_image)
{
    const int width = 100;
    const int height = 100;
    cv::Mat_<int> src(height, width);

    for(int y=0; y<src.rows; ++y) {
        for(int x=0; x<src.cols; ++x) {
            src(y, x) = x*y;
        }
    }
    std::pair<int, int> result = cvutil::minmax_element_image(src);
    int min_result = cvutil::min_element_image(src);
    int max_result = cvutil::max_element_image(src);
    ASSERT_EQ(result.first, 0);
    ASSERT_EQ(result.second, (width - 1)*(height - 1));
    ASSERT_EQ(result.first, min_result);
    ASSERT_EQ(result.second, max_result);

    int min = std::numeric_limits<int>::max();
    int max = std::numeric_limits<int>::min();
    for(int y=0; y<src.rows; ++y) {
        for(int x=0; x<src.cols; ++x) {
            int value = rand();
            if(value < min) min = value;
            if(value > max) max = value;
            src(y, x) = value;
        }
    }
    result = cvutil::minmax_element_image(src);
    min_result = cvutil::min_element_image(src);
    max_result = cvutil::max_element_image(src);
    ASSERT_EQ(result.first, min);
    ASSERT_EQ(result.second, max);
    ASSERT_EQ(result.first, min_result);
    ASSERT_EQ(result.second, max_result);
}
