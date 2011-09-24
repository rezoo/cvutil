#include <gtest/gtest.h>

#include <functional>
#include <algorithm>
#include <opencv2/core/core.hpp>

#include <cvutil/algorithm/integrate.hpp>

TEST(algorithmTest, integrate_image)
{
    const int width = 100;
    const int height = 100;
    cv::Mat_<int> src(height, width);
    cv::Mat_<int> dst1(height, width);
    cv::Mat_<int> dst2(height, width);

    for(int y=0; y<src.rows; ++y) {
        for(int x=0; x<src.cols; ++x) {
            src(y, x) = 1;
        }
    }

    cvutil::integrate_image(src, dst1, std::plus<int>());

    for(int y=0; y<dst2.rows; ++y) {
        for(int x=0; x<dst2.cols; ++x) {
            dst2(y, x) = (x+1)*(y+1);
        }
    }

    ASSERT_TRUE(std::equal(dst1.begin(), dst1.end(), dst2.begin()));
}
