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

namespace cvutil {

template<typename SrcType, typename DstType>
void x_gradient(const cv::Mat_<SrcType>& src, cv::Mat_<DstType>& dst) {
    assert(src.size == dst.size);
    for(int y=0; y<src.rows; ++y) {
        const SrcType* src_x = &src(y, 0);
        DstType* dst_x = &dst(y, 0);
        for(int x=1; x<(src.cols-1); ++x) {
            dst_x[x] = (src_x[x+1] - src_x[x-1])/2;
        }
    }
}

template<typename SrcType, typename DstType>
void y_gradient(const cv::Mat_<SrcType>& src, cv::Mat_<DstType>& dst) {
    assert(src.size == dst.size);
    const std::size_t y_step = src.stepT();
    for(int y=1; y<(src.rows-1); ++y) {
        const SrcType* src_x = &src(y, 0);
        DstType* dst_x = &dst(y, 0);
        for(int x=0; x<src.cols; ++x) {
            dst_x[x] = (src_x[x+y_step] - src_x[x-y_step])/2;
        }
    }
}

template<typename SrcType, typename DstType>
void gradient(const cv::Mat_<SrcType>& src,
              cv::Mat_<DstType>& dst_x,
              cv::Mat_<DstType>& dst_y) {
    x_gradient(src, dst_x);
    y_gradient(src, dst_y);
}

template<typename SrcType, typename DstType>
void gradient(const cv::Mat_<SrcType>& src,
              cv::Mat_<cv::Point_<DstType> >& dst) {
    assert(src.size == dst.size);
    const std::size_t y_step = src.stepT();
    for(int y=1; y<(src.rows-1); ++y) {
        const SrcType* src_x = &src(y, 0);
        DstType* dst_x = &dst(y, 0);
        for(int x=1; x<(src.cols-1); ++x) {
            dst_x[x] = Point_<DstType>(
                static_cast<DstType>(src_x[x+1] - src_x[x-1])/2,
                static_cast<DstType>(src_x[x+y_step] - src_x[x-y_step])/2);
        }
    }
}

} // namespace cvutil
