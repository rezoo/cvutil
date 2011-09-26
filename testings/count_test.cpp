#include <gtest/gtest.h>

#include <functional>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <cvutil/algorithm/count.hpp>

TEST(algorithmTest, count_image)
{
    const int width = 10;
    const int height = 10;
    cv::Mat_<int> src(height, width);
    for(int y=0; y<src.rows; ++y) {
        for(int x=0; x<src.cols; ++x) {
            src(y, x) = 5;
        }
    }

    int result = cvutil::count_image(src, 5);
    ASSERT_EQ(result, width*height);

    for(int y=0; y<src.rows; ++y) {
        for(int x=0; x<src.cols; ++x) {
            src(y, x) = x + y;
        }
    }
    result = cvutil::count_image_if(src, [](int x){ return x < 5; });
    ASSERT_EQ(result, 15);
}

