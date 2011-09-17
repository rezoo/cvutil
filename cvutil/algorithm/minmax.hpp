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
#include <algorithm>
#include <utility>

namespace cvutil {
namespace detail {

template<typename T, typename BinaryPredicate>
T get_element_image(const cv::Mat_<T>& src,
                    BinaryPredicate comp) {
    T result = src(0, 0);
    for(int y=0; y<src.rows; ++y) {
        const T* src_x = src[y];
        for(int x=0; x<src.cols; ++x) {
            if(comp(src_x[x], result))
                result = src_x[x];
        }
    }
    return result;
}

} // namespace detail

template<typename T>
T min_element_image(const cv::Mat_<T>& src) {
    return detail::get_element_image(src, std::less<T>());
}

template<typename T>
T max_element_image(const cv::Mat_<T>& src) {
    return detail::get_element_image(src, std::greater<T>());
}

template<typename T, typename BinaryPredicate>
std::pair<T, T> minmax_element_image(const cv::Mat_<T>& src,
                                     BinaryPredicate comp) {
    T min_result = src(0, 0);
    T max_result = min_result;
    for(int y=0; y<src.rows; ++y) {
        const T* src_x = src[y];
        for(int x=0; x<src.cols; ++x) {
            if(comp(src_x[x], min_result))
                min_result = src_x[x];
            if(comp(max_result, src_x[x]))
                max_result = src_x[x];
        }
    }
    return std::make_pair(min_result, max_result);
}

template<typename T>
std::pair<T, T> minmax_element_image(const cv::Mat_<T>& src) {
    return minmax_element_image(src, std::less<T>());
}

} // namespace cvutil
