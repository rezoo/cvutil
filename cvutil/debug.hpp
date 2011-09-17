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

#include <iostream>
#include <opencv2/core/core.hpp>
#include <utility>
#include "algorithm/minmax.hpp"
#include "algorithm/transform.hpp"

namespace cvutil {
namespace detail {

template<typename T>
struct show_image_functor {
public:
    show_image_functor(T min, T max)
    : min_(min), delta_(max - min) {}

    T operator()(T x) { return (x - min_)/delta_; }
private:
    T min_;
    T delta_;
};

}

template<typename T>
void show_image(const cv::Mat_<T>& img, const std::string& title) {
    cv::Mat_<T> tmp_img(img.rows, img.cols);
    std::pair<T, T> m = cvutil::minmax_element_image(img);
    const double delta = m.second - m.first;

    if(m.first == m.second) return;

    std::cout << title << " min:" << m.first
              << " max:" << m.second << std::endl;
    cvutil::transform_image(
        img, tmp_img,
        detail::show_image_functor<T>(m.first, m.second));
    cv::imshow(title, tmp_img);
}

} // namespace cvutil
