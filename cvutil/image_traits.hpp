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

#include <numeric>

namespace cvutil {

template<typename T>
struct image_traits {};

template<>
struct image_traits<unsigned char> {
    typedef unsigned char value_type;
    typedef int cast_type;
    typedef uchar convertion_type;
    static value_type max() { return std::numeric_limits<value_type>::max(); }
    static value_type min() { return std::numeric_limits<value_type>::min(); }
    static value_type zero() { return 0; }
    static bool need_to_convert() { return false; }
};

template<>
struct image_traits<char> {
    typedef char value_type;
    typedef int cast_type;
    typedef uchar convertion_type;
    static value_type max() { return std::numeric_limits<value_type>::max(); }
    static value_type min() { return std::numeric_limits<value_type>::min(); }
    static value_type zero() { return 0; }
    static bool need_to_convert() { return true; }
};

template<>
struct image_traits<float> {
    typedef float value_type;
    typedef float cast_type;
    typedef float convertion_type;
    static value_type max() { return 1.0f; }
    static value_type min() { return 0.0f; }
    static value_type zero() { return 0.0f; }
    static bool need_to_convert() { return false; }
};

template<>
struct image_traits<double> {
    typedef double value_type;
    typedef double cast_type;
    typedef double convertion_type;
    static value_type max() { return 1.0; }
    static value_type min() { return 0.0; }
    static value_type zero() { return 0.0; }
    static bool need_to_convert() { return false; }
};

} // namespace cvutil
