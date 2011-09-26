#include <gtest/gtest.h>

#include <stdlib.h>
#include <functional>
#include <algorithm>
#include <utility>
#include <opencv2/core/core.hpp>

#include <cvutil/algorithm/copy.hpp>

TEST(algorithmTest, copy_image)
{
    const int width = 100;
    const int height = 100;
    cv::Mat_<int> src(height, width);
    cv::Mat_<int> dst(height, width);

    for(int y=0; y<src.rows; ++y) {
        for(int x=0; x<src.cols; ++x) {
            src(y, x) = rand();
        }
    }
    cvutil::copy_image(src, dst);
    ASSERT_TRUE(std::equal(src.begin(), src.end(), dst.begin()));
}
