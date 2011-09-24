#include <gtest/gtest.h>

#include <functional>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <cvutil/algorithm/reduce.hpp>

TEST(algorithmTest, reduce_image)
{
    cv::Mat_<int> src(10, 10);
    for(int y=0; y<src.cols; ++y) {
        for(int x=0; x<src.rows; ++x) {
            src(y, x) = y*src.cols + x;
        }
    }
    int result1 = cvutil::reduce_image(src, (int)0, std::plus<int>());
    int result2 = std::accumulate(src.begin(), src.end(), (int)0);
    ASSERT_EQ(result1, result2);

    result1 = cvutil::reduce_image(src, (int)0);
    ASSERT_EQ(result1, result2);
    //result1 = cvutil::reduce_image(src);
    //ASSERT_EQ(result1, result2);

    result1 = cvutil::reduce_image(
        src, (int)1, std::multiplies<int>());
    result2 = std::accumulate(
        src.begin(), src.end(), (int)1, std::multiplies<int>());
    ASSERT_EQ(result1, result2);
}
