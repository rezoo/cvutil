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

template<typename DstType, typename Function>
void generate_image(cv::Mat_<DstType>& dst, Function f) {
    #pragma omp parallel for schedule(dynamic)
    for(int y=0; y<dst.rows; ++y) {
        DstType* dst_x = dst[y];
        for(int x=0; x<dst.cols; ++x) {
            dst_x[x] = f();
        }
    }
}

template<typename DstType, typename CoordinateFunction>
void generate_image_xy(cv::Mat_<DstType>& dst,
                       CoordinateFunction f) {
    #pragma omp parallel for schedule(dynamic)
    for(int y=0; y<dst.rows; ++y) {
        DstType* dst_x = dst[y];
        for(int x=0; x<dst.cols; ++x) {
            dst_x[x] = f(x, y);
        }
    }
}

template<typename DstType, typename CoordinateFunction>
void generate_image_uv(cv::Mat_<DstType>& dst,
                       CoordinateFunction f) {
    #pragma omp parallel for schedule(dynamic)
    const int cx = dst.cols / 2;
    const int cy = dst.rows / 2;
    for(int y=0; y<dst.rows; ++y) {
        DstType* dst_x = dst[y];
        for(int x=0; x<dst.cols; ++x) {
            dst_x[x] = f(x - cx, cy - y);
        }
    }
}

} // namespace cvutil
