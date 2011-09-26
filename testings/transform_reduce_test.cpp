#include <gtest/gtest.h>

#include <functional>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <cvutil/algorithm/reduce.hpp>
#include <cvutil/algorithm/transform_reduce.hpp>

TEST(algorithmTest, transform_reduce_image)
{
    const int width = 10;
    const int height = 10;
    cv::Mat_<int> src1(height, width);
    for(int y=0; y<src1.rows; ++y) {
        for(int x=0; x<src1.cols; ++x) {
            src1(y, x) = y*src1.cols + x;
        }
    }
    cv::Mat_<int> src2(height, width);
    for(int y=0; y<src2.rows; ++y) {
        for(int x=0; x<src2.cols; ++x) {
            src2(y, x) = x + y;
        }
    }
    cv::Mat_<int> src3(height, width);
    for(int y=0; y<src3.rows; ++y) {
        for(int x=0; x<src3.cols; ++x) {
            src3(y, x) = x*y;
        }
    }

    int result1 = cvutil::transform_reduce_image(
        src1, [](int x){ return x*2; }, (int)0, std::plus<int>());
    int result2 = std::accumulate(src1.begin(), src1.end(), (int)0);
    ASSERT_EQ(result1, 2*result2);

    result1 = cvutil::transform_reduce_image(
        src1, src2, std::plus<int>(), (int)0, std::plus<int>());
    result2 = cvutil::reduce_image(src1) + cvutil::reduce_image(src2);
    ASSERT_EQ(result1, result2);

    result1 = cvutil::transform_reduce_image(
        src1, src2, src3,
        [](int x, int y, int z){ return x+y+z; },
        (int)0, std::plus<int>());
    result2 = cvutil::reduce_image(src1) +
              cvutil::reduce_image(src2) +
              cvutil::reduce_image(src3);
    ASSERT_EQ(result1, result2);
}

