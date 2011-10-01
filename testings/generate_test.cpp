#include <gtest/gtest.h>

#include <algorithm>
#include <opencv2/core/core.hpp>

#include <cvutil/algorithm/generate.hpp>

TEST(algorithmTest, generate_image)
{
    const int width = 100;
    const int height = 100;
    cv::Mat_<int> src1(height, width);
    cv::Mat_<int> src2(height, width);

    for(int y=0; y<src1.rows; ++y) {
        for(int x=0; x<src1.cols; ++x) {
            src1(y, x) = 5;
        }
    }
    cvutil::generate_image(src2, [](){ return 5; });
    ASSERT_TRUE(std::equal(src1.begin(), src1.end(), src2.begin()));
}

TEST(algorithmTest, generate_image_xy)
{
    const int width = 100;
    const int height = 100;
    cv::Mat_<int> src1(height, width);
    cv::Mat_<int> src2(height, width);

    for(int y=0; y<src1.rows; ++y) {
        for(int x=0; x<src1.cols; ++x) {
            src1(y, x) = x + y;
        }
    }
    cvutil::generate_image_xy(src2, [](int x, int y){ return x + y; });
    ASSERT_TRUE(std::equal(src1.begin(), src1.end(), src2.begin()));
}

TEST(algorithmTest, generate_image_uv)
{
    const int width = 100;
    const int height = 100;
    cv::Mat_<int> src1(height, width);
    cv::Mat_<int> src2(height, width);

    for(int y=0; y<src1.rows; ++y) {
        for(int x=0; x<src1.cols; ++x) {
            int u = x - width/2;
            int v = -(y - height/2);
            src1(y, x) = u*v;
        }
    }
    cvutil::generate_image_uv(src2, [](int u, int v){ return u * v; });
    ASSERT_TRUE(std::equal(src1.begin(), src1.end(), src2.begin()));
}
