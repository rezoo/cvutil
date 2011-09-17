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

#include <opencv2/opencv.hpp>
#include <boost/algorithm/minmax.hpp>

namespace cvutil {

template<typename SrcType, typename DstType>
cv::Vec<DstType, 3> hsv_to_bgr(const cv::Vec<SrcType, 3>& x) {
    const DstType hue = static_cast<DstType>(x[0]);
    const DstType saturation = static_cast<DstType>(x[1]);
    const DstType value = static_cast<DstType>(x[2]);

    if(x[1] == 0)
        return cv::Vec<DstType, 3>(value, value, value);

    const DstType h = std::fmod<DstType>(hue/60, 6);
    const int h_i = static_cast<int>(x[0]) % 6;
    const DstType f = h - static_cast<DstType>(h_i);
    const DstType p = value*(1 - saturation);
    const DstType q = value*(1 - f*saturation);
    const DstType t = value*(1 - (1-f)*saturation);

    if(h_i == 0)
        return cv::Vec<DstType, 3>(value, t, p);
    else if(h_i == 1)
        return cv::Vec<DstType, 3>(q, value, p);
    else if(h_i == 2)
        return cv::Vec<DstType, 3>(p, value, t);
    else if(h_i == 3)
        return cv::Vec<DstType, 3>(p, q, value);
    else if(h_i == 4)
        return cv::Vec<DstType, 3>(t, p, value);
    else
        return cv::Vec<DstType, 3>(value, p, q);
}

template<typename SrcType, typename DstType>
cv::Vec<DstType, 3> bgr_to_hsv(const cv::Vec<SrcType, 3>& x) {
    const boost::tuple<SrcType, SrcType>
        tmp_minmax = boost::minmax(x[0], x[1]);
    const SrcType i_min = std::min(boost::get<0>(tmp_minmax), x[2]);
    const SrcType i_max = std::max(boost::get<1>(tmp_minmax), x[2]);

    DstType hue;
    const DstType delta_i = static_cast<DstType>(i_max - i_min);
    if(x[2] == i_max) {
        hue = 60*(x[0]-x[1])/delta_i;
    } else if(x[1] == i_max) {
        hue = 60*(2 + (x[2]-x[0])/delta_i);
    } else {
        hue = 60*(4 + (x[1]-x[2])/delta_i);
    }
    const DstType dsttype_i_max = static_cast<DstType>(i_max);

    return cv::Vec<DstType, 3>(hue, delta_i/dsttype_i_max, dsttype_i_max);
}

} // namespace cvutil
