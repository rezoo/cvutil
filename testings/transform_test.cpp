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

TEST(algorithmTest, transform_image_xy)
{
    const int width = 10;
    const int height = 10;
    cv::Mat_<int> src1(height, width);
    for(int y=0; y<src1.rows; ++y) {
        for(int x=0; x<src1.cols; ++x) {
            src1(y, x) = y * x;
        }
    }
    cv::Mat_<int> src2(height, width);
    for(int y=0; y<src2.rows; ++y) {
        for(int x=0; x<src2.cols; ++x) {
            src2(y, x) = 2*x + 3*y;
        }
    }

    cv::Mat_<int> dst1(height, width);
    cvutil::transform_image_xy(src1, dst1,
        [](int p, int x, int y){ return p + x + 3*y; });
    cv::Mat_<int> dst2(height, width);
    for(int y=0; y<dst2.rows; ++y) {
        for(int x=0; x<dst2.cols; ++x) {
            dst2(y, x) = x*y + x + 3*y;
        }
    }
    ASSERT_TRUE(std::equal(dst1.begin(), dst1.end(), dst2.begin()));

    cvutil::transform_image_xy(src1, src2, dst1,
        [](int p1, int p2, int x, int y){ return p1*p2 + x + 5*y; });
    for(int y=0; y<dst2.rows; ++y) {
        for(int x=0; x<dst2.cols; ++x) {
            dst2(y, x) = x*y*(2*x + 3*y) + x + 5*y;
        }
    }
    ASSERT_TRUE(std::equal(dst1.begin(), dst1.end(), dst2.begin()));
}

TEST(algorithmTest, transform_image_uv)
{
    const int width = 10;
    const int height = 10;
    cv::Mat_<int> src1(height, width);
    for(int y=0; y<src1.rows; ++y) {
        for(int x=0; x<src1.cols; ++x) {
            src1(y, x) = x + y;
        }
    }
    cv::Mat_<int> src2(height, width);
    for(int y=0; y<src2.rows; ++y) {
        for(int x=0; x<src2.cols; ++x) {
            src2(y, x) = 2*x + 3*y;
        }
    }

    cv::Mat_<int> dst(height, width);
    cvutil::transform_image_uv(src1, dst,
        [](int p, int u, int v){ return p + u + 3*v; });
    ASSERT_EQ(dst(0, 0), 0 + 0 - 5 + 3*5);
    ASSERT_EQ(dst(9, 0), 0 + 9 - 5 - 3*4);
    ASSERT_EQ(dst(0, 9), 9 + 0 + 4 + 3*5);
    ASSERT_EQ(dst(9, 9), 9 + 9 + 4 - 3*4);

    cvutil::transform_image_uv(src1, src2, dst,
        [](int p1, int p2, int u, int v){ return p1*p2 + 3*u + v; });
    ASSERT_EQ(dst(0, 0), (0+0)*(2*0 + 3*0) - 3*5 + 5);
    ASSERT_EQ(dst(9, 0), (0+9)*(2*0 + 3*9) - 3*5 - 4);
    ASSERT_EQ(dst(0, 9), (9+0)*(2*9 + 3*0) + 3*4 + 5);
    ASSERT_EQ(dst(9, 9), (9+9)*(2*9 + 3*9) + 3*4 - 4);
}
