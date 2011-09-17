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
         typename DstType,
         typename UnaryFunction>
void transform_image(const cv::Mat_<SrcType>& src,
                     cv::Mat_<DstType>& dst,
                     UnaryFunction f) {
    assert(src.size == dst.size);
    #pragma omp parallel for schedule(dynamic)
    for(int y=0; y<dst.rows; ++y) {
        const SrcType* src_x = src[y];
        DstType* dst_x = dst[y];
        for(int x=0; x<dst.cols; ++x) {
            dst_x[x] = f(src_x[x]);
        }
    }
}

template<typename SrcType1,
         typename SrcType2,
         typename DstType,
         typename BinaryFunction>
void transform_image(const cv::Mat_<SrcType1>& src1,
                     const cv::Mat_<SrcType2>& src2,
                     cv::Mat_<DstType>& dst,
                     BinaryFunction f) {
    assert(src1.size == dst.size);
    assert(src2.size == dst.size);
    #pragma omp parallel for schedule(dynamic)
    for(int y=0; y<dst.rows; ++y) {
        const SrcType1* src1_x = src1[y];
        const SrcType2* src2_x = src2[y];
        DstType* dst_x = dst.ptr<DstType>();
        for(int x=0; x<dst.cols; ++x) {
            dst_x[x] = f(src1_x[x], src2_x[x]);
        }
    }
}

template<typename SrcType1,
         typename SrcType2,
         typename SrcType3,
         typename DstType,
         typename TernaryFunction>
void transform_image(const cv::Mat_<SrcType1>& src1,
                     const cv::Mat_<SrcType2>& src2,
                     const cv::Mat_<SrcType3>& src3,
                     cv::Mat_<DstType>& dst,
                     TernaryFunction f) {
    assert(src1.size == dst.size);
    assert(src2.size == dst.size);
    assert(src3.size == dst.size);
    #pragma omp parallel for schedule(dynamic)
    for(int y=0; y<dst.rows; ++y) {
        const SrcType1* src1_x = src1[y];
        const SrcType2* src2_x = src2[y];
        const SrcType3* src3_x = src3[y];
        DstType* dst_x = dst[y];
        for(int x=0; x<dst.cols; ++x) {
            dst_x[x] = f(src1_x[x], src2_x[x], src3_x[x]);
        }
    }
}

template<typename SrcType,
         typename DstType,
         typename CoordinateFunction>
void transform_image_xy(const cv::Mat_<SrcType>& src,
                        cv::Mat_<DstType>& dst,
                        CoordinateFunction f) {
    assert(src.size == dst.size);
    #pragma omp parallel for schedule(dynamic)
    for(int y=0; y<dst.rows; ++y) {
        const SrcType* src_x = src[y];
        DstType* dst_x = dst[y];
        for(int x=0; x<dst.cols; ++x) {
            dst_x[x] = f(src_x[x], x, y);
        }
    }
}

template<typename SrcType1,
         typename SrcType2,
         typename DstType,
         typename CoordinateFunction>
void transform_image_xy(const cv::Mat_<SrcType1>& src1,
                        const cv::Mat_<SrcType2>& src2,
                        cv::Mat_<DstType>& dst,
                        CoordinateFunction f) {
    assert(src1.size == dst.size);
    assert(src2.size == dst.size);
    #pragma omp parallel for schedule(dynamic)
    for(int y=0; y<dst.rows; ++y) {
        const SrcType1* src1_x = src1[y];
        const SrcType2* src2_x = src2[y];
        DstType* dst_x = dst[y];
        for(int x=0; x<dst.cols; ++x) {
            dst_x[x] = f(src1_x[x], src2_x[x], x, y);
        }
    }
}

template<typename SrcType,
         typename DstType,
         typename CoordinateFunction>
void transform_image_uv(const cv::Mat_<SrcType>& src,
                        cv::Mat_<DstType>& dst,
                        CoordinateFunction f) {
    assert(src.size == dst.size);
    const int cx = dst.cols / 2;
    const int cy = dst.rows / 2;
    #pragma omp parallel for schedule(dynamic)
    for(int y=0; y<dst.rows; ++y) {
        const SrcType* src_x = src[y];
        DstType* dst_x = dst[y];
        for(int x=0; x<dst.cols; ++x) {
            dst_x[x] = f(src_x[x], x - cx, y - cy);
        }
    }
}

template<typename SrcType1,
         typename SrcType2,
         typename DstType,
         typename CoordinateFunction>
void transform_image_uv(const cv::Mat_<SrcType1>& src1,
                        const cv::Mat_<SrcType2>& src2,
                        cv::Mat_<DstType>& dst,
                        CoordinateFunction f) {
    assert(src1.size == dst.size);
    assert(src2.size == dst.size);
    const int cx = dst.cols / 2;
    const int cy = dst.rows / 2;
    #pragma omp parallel for schedule(dynamic)
    for(int y=0; y<dst.rows; ++y) {
        const SrcType1* src1_x = src1[y];
        const SrcType2* src2_x = src2[y];
        DstType* dst_x = dst[y];
        for(int x=0; x<dst.cols; ++x) {
            dst_x[x] = f(src1_x[x], src2_x[x], x - cx, y - xy);
        }
    }
}

} // namespace cvutil
