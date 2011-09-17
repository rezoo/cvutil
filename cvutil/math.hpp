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

#include <cmath>

namespace cvutil {

template<typename T>
inline T sinh_3rd(T x) {
    return x*(1 + x*x/6);
}

template<typename T>
inline T sinh_5th(T x) {
    const T x2 = x*x;
    return x*(1 + x2/6*(1 + x2/20));
}

template<typename T>
inline T sinh_7th(T x) {
    const T x2 = x*x;
    return x*(1 + x2/6*(1 + x2/20*(1 + x2/42)));
}

template<typename T>
inline T cosh_2nd(T x) {
    return 1 + x*x/2;
}

template<typename T>
inline T cosh_4th(T x) {
    const T x2 = x*x;
    return 1 + x2/2*(1 + x2/12);
}

template<typename T>
inline T cosh_6th(T x) {
    const T x2 = x*x;
    return 1 + x2/2*(1 + x2/12*(1 + x2/30));
}

template<typename T>
inline T tanh_4th(T x) {
    const T xa = std::abs(x);
    const T x2 = xa*xa;

    const T a = (T)1.22773;
    const T result = 1 - 1/(1 + xa + x2*(1 + a*x2));
    return (x > 0 ? result : -result);
}

template<typename T>
inline T tanh_7th(T x) {
    const T xa = std::abs(x);
    const T x2 = xa*xa;
    const T x3 = xa*x2;
    const T x4 = x2*x2;
    const T x7 = x3*x4;

    const T a = (T)0.525888;
    const T b = (T)0.616127;
    const T c = (T)0.0526317;
    const T result = 1 - 1/(1 + xa + x2 + a*x3 + b*x4 + c*x7);
    return (x > 0 ? result : -result);
}

template<typename T>
inline T sigmoid(T a, T x) {
    return 1/(1 + std::exp(-a*x));
}

template<typename T>
inline T sigmoid_4th(T a, T x) {
    return (tanh_4th(a*x) + 1)/2;
}

template<typename T>
inline T sigmoid_7th(T a, T x) {
    return (tanh_7th(a*x) + 1)/2;
}

template<typename T>
inline T fermi_distribution(T temperature,
                            T chemical_potential,
                            T x) {
    return 1/(1 + std::exp(1/temperature*(x - chemical_potential)));
}

} // namespace cvutil
