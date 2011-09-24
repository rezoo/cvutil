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

#include <opencv2/core/core.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cvutil {

template<typename SrcType,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
ForwardIterator transform_image_if(const cv::Mat_<SrcType>& src,
                                   ForwardIterator result,
                                   UnaryFunction f,
                                   Predicate pred) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for(int y=0; y<src.rows; ++y) {
        const SrcType* src_x = src[y];
        for(int x=0; x<src.cols; ++x) {
            if(pred(src_x[x])) {
                *result = f(src_x[x]);
                ++result;
            }
        }
    }
    return result;
}

template<typename SrcType1,
         typename SrcType2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
ForwardIterator transform_image_if(const cv::Mat_<SrcType1>& src1,
                                   const cv::Mat_<SrcType2>& src2,
                                   ForwardIterator result,
                                   UnaryFunction f,
                                   Predicate pred) {
    assert(src1.size == src2.size);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for(int y=0; y<src1.rows; ++y) {
        const SrcType1* src1_x = src1[y];
        const SrcType2* src2_x = src2[y];
        for(int x=0; x<src1.cols; ++x) {
            if(pred(src1_x[x], src2_x[x])) {
                *result = f(src1_x[x], src2_x[x]);
                ++result;
            }
        }
    }
    return result;
}

} // namespace cvutil
