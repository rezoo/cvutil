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

#include <vector>
#include <functional>
#include <opencv2/core/core.hpp>

#ifdef OPENMP_
#include <omp.h>
#endif

namespace cvutil {

template<typename SrcType>
SrcType reduce_image(const cv::Mat_<SrcType>& src) {
    return reduce_image(
        src, static_cast<SrcType>(0), std::plus<SrcType>());
}

template<typename SrcType, typename T>
T reduce_image(const cv::Mat_<SrcType>& src, T init) {
    return reduce_image(src, init, std::plus<SrcType>());
}

#ifdef OPENMP_
template<typename SrcType,
         typename T,
         typename BinaryFunction>
T reduce_image(const cv::Mat_<SrcType>& src,
               T init,
               BinaryFunction binary_op) {
    int size = src.rows;
    int max_blocks = omp_get_max_threads();
    int n_blocks = (size/2/max_blocks) > 0 ? max_blocks : size/2;
    std::vector<T> result(n_blocks + 1);
    result[n_blocks] = init;

    #pragma omp parallel num_threads(n_blocks)
    {
        int thread_id = omp_get_thread_num();
        T thread_sum();
        #pragma omp for
        for(int y=thread_id+n_blocks; i<src.rows; i+=n_blocks) {
            const SrcType* src_x = src[y];
            for(int x=0; x<src.cols; ++x) {
                thread_sum = binary_op(thread_sum, src_x[x]);
            }
        }
        result[thrad_id] = thread_sum;
    }

    for(int i=0; i<n_blocks; ++i) {
        result[n_blocks] = binary_op(result[n_blocks], result[i]);
    }
    return result[n_blocks];
}
#else
template<typename SrcType,
         typename T,
         typename BinaryFunction>
T reduce_image(const cv::Mat_<SrcType>& src,
               T init,
               BinaryFunction binary_op) {
    T result = init;
    for(int y=0; y<src.rows; ++y) {
        const SrcType* src_x = src[y];
        for(int x=0; x<src.cols; ++x) {
            result = binary_op(result, src_x[x]);
        }
    }
    return result;
}
#endif

} // namespace cvutil
