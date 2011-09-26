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
#include "transform_reduce.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cvutil {
namespace detail {

template<typename UnaryPredicate>
struct count_if_functor {
public:
    count_if_functor(UnaryPredicate f): unary_pred_(f) {};

    template<typename T>
    std::size_t operator()(const T& x) {
        return unary_pred_(x) ? 1 : 0;
    }
private:
    UnaryPredicate unary_pred_;
};

} // namespace detail

template<typename SrcType, typename UnaryPredicate>
std::size_t count_image_if(const cv::Mat_<SrcType>& src,
                           UnaryPredicate unary_pred) {
    return transform_reduce_image(
        src,
        detail::count_if_functor<UnaryPredicate>(unary_pred),
        (std::size_t)0, 
        std::plus<std::size_t>());
}

template<typename SrcType>
std::size_t count_image(const cv::Mat_<SrcType>& src, SrcType value) {
    return count_image_if(
        src, std::bind1st(std::equal_to<SrcType>(), value));
}

} // namespace cvutil
