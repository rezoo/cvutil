#include <gtest/gtest.h>

#include <functional>
#include <algorithm>
#include <opencv2/core/core.hpp>

#include <cvutil/algorithm/transform.hpp>

TEST(algorithmTest, transform_image)
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
            src2(y, x) = x*src2.cols + y;
        }
    }
    cv::Mat_<int> src3(height, width);
    for(int y=0; y<src3.rows; ++y) {
        for(int x=0; x<src3.cols; ++x) {
            src3(y, x) = x*y;
        }
    }
    cv::Mat_<int> dst1(height, width);
    cv::Mat_<int> dst2(height, width);

    cvutil::transform_image(src1, dst1, [](int x){ return x*2; });
    for(int y=0; y<dst2.rows; ++y) {
        for(int x=0; x<dst2.cols; ++x) {
            dst2(y, x) = src1(y, x)*2;
        }
    }
    ASSERT_TRUE(std::equal(dst2.begin(), dst2.end(), dst2.begin()));
    
    cvutil::transform_image(src1, src2, dst1,
        [](int x, int y){ return x*2 + y*2; });
    for(int y=0; y<dst2.rows; ++y) {
        for(int x=0; x<dst2.cols; ++x) {
            dst2(y, x) = src1(y, x)*2 + src2(y, x)*2;
        }
    }
    ASSERT_TRUE(std::equal(dst2.begin(), dst2.end(), dst2.begin()));

    cvutil::transform_image(src1, src2, src3, dst1,
        [](int x, int y, int z){ return x + y + z; });
    for(int y=0; y<dst2.rows; ++y) {
        for(int x=0; x<dst2.cols; ++x) {
            dst2(y, x) = src1(y, x) + src2(y, x) + src3(y, x);
        }
    }
    ASSERT_TRUE(std::equal(dst2.begin(), dst2.end(), dst2.begin()));
}
