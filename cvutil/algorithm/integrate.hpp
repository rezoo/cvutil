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

template<typename T,
         typename BinaryFunction>
void integrate_image(cv::Mat_<T>& src,
                     cv::Mat_<T>& dst,
                     BinaryFunction f) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for(int y=0; y<src.rows; ++y) {
        dst(y, 0) = src(y, 0);
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for(int y=0; y<src.rows; ++y) {
        T* src_x = src[y];
        T* dst_x = dst[y];
        for(int x=1; x<src.cols; ++x) {
            dst_x[x] = f(dst_x[x-1], src_x[x]);
        }
    }

    const std::size_t step = src.step.p[0];
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for(int x=0; x<src.cols; ++x) {
        uchar* dst_py = (uchar*)&dst(1, x) - step;
        uchar* dst_y  = (uchar*)&dst(1, x);
        for(int y=1; y<src.rows; ++y, dst_py += step, dst_y += step) {
            *(T*)dst_y = f(*(T*)dst_py, *(T*)dst_y);
        }
    }
}

} // namespace cvutil
