/* The MIT License
 * 
 * Copyright (c) 2011 Masaki Saito <rezoolab@gmail.com>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <functional>
#include <opencv2/core/core.hpp>

namespace cvutil {

template<typename SrcType1,
         typename SrcType2,
         typename T,
         typename BinaryFunction1,
         typename BinaryFunction2>
T inner_product_image(const cv::Mat_<SrcType1>& src1,
                      const cv::Mat_<SrcType2>& src2,
                      T init,
                      BinaryFunction1 binary_op1,
                      BinaryFunction2 binary_op2) {
    assert(src1.size == src2.size);
    T result = init;
    for(int y=0; y<src1.rows; ++y) {
        const SrcType1* src1_x = src1[y];
        const SrcType2* src2_x = src2[y];
        for(int x=0; x<src1.cols; ++x) {
            result = binary_op1(result, binary_op2(src1_x[x], src2_x[x]));
        }
    }
    return result;
}

template<typename SrcType1,
         typename SrcType2,
         typename T>
T inner_product_image(const cv::Mat_<SrcType1>& src1,
                      const cv::Mat_<SrcType2>& src2,
                      T init) {
    return inner_product_image(
        src1, src2,
        init,
        std::plus<SrcType1>(), std::multiplies<SrcType1>());
}

template<typename SrcType1,
         typename SrcType2>
SrcType1 inner_product_image(const cv::Mat_<SrcType1>& src1,
                             const cv::Mat_<SrcType2>& src2) {
    return inner_product_image(
        src1, src2,
        static_cast<SrcType1>(0),
        std::plus<SrcType1>(), std::multiplies<SrcType1>());
}

} // namespace cvutil
