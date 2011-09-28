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

#include <utility>
#include <vector>
#include <functional>
#include <opencv2/core/core.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cvutil {

template<typename SrcType, typename BinaryPredicate>
SrcType min_element_image(const cv::Mat_<SrcType>& src,
                          BinaryPredicate comp) {
#ifdef _OPENMP
    const int size = src.rows;
    const int max_blocks = omp_get_max_threads();
    const int n_blocks = (size/max_blocks) > 0 ? max_blocks : size;
    std::vector<SrcType> results(n_blocks);

    #pragma omp parallel num_threads(n_blocks)
    {
        int thread_id = omp_get_thread_num();
        SrcType thread_result = src(thread_id, 0);
        for(int y=thread_id; y<src.rows; y+=n_blocks) {
            const SrcType* src_x = src[y];
            for(int x=0; x<src.cols; ++x) {
                if(comp(src_x[x], thread_result))
                    thread_result = src_x[x];
            }
        }
        results[thread_id] = thread_result;
    }

    if(n_blocks > 1) {
        for(int i=1; i<n_blocks; ++i) {
            if(comp(results[i], results[0]))
                results[0] = results[i];
        }
        return results[0];
    } else {
        return results[0];
    }
#else
    T result = src(0, 0);
    for(int y=0; y<src.rows; ++y) {
        const T* src_x = src[y];
        for(int x=0; x<src.cols; ++x) {
            if(comp(src_x[x], result))
                result = src_x[x];
        }
    }
    return result;
#endif
}

template<typename SrcType>
SrcType min_element_image(const cv::Mat_<SrcType>& src) {
    return min_element_image(src, std::less<SrcType>());
}

template<typename SrcType, typename BinaryPredicate>
SrcType max_element_image(const cv::Mat_<SrcType>& src,
                          BinaryPredicate comp) {
    return min_element_image(src, comp);
}

template<typename SrcType>
SrcType max_element_image(const cv::Mat_<SrcType>& src) {
    return max_element_image(src, std::greater<SrcType>());
}

template<typename SrcType, typename BinaryPredicate>
std::pair<SrcType, SrcType> minmax_element_image(const cv::Mat_<SrcType>& src,
                                                 BinaryPredicate comp) {
#ifdef _OPENMP
    const int size = src.rows;
    const int max_blocks = omp_get_max_threads();
    const int n_blocks = (size/max_blocks) > 0 ? max_blocks : size;
    std::vector<std::pair<SrcType, SrcType> > results(n_blocks);

    #pragma omp parallel num_threads(n_blocks)
    {
        int thread_id = omp_get_thread_num();
        SrcType thread_min = src(thread_id, 0);
        SrcType thread_max = thread_min;
        for(int y=thread_id; y<src.rows; y+=n_blocks) {
            const SrcType* src_x = src[y];
            for(int x=0; x<src.cols; ++x) {
                if(comp(src_x[x], thread_min))
                    thread_min = src_x[x];
                if(comp(thread_max, src_x[x]))
                    thread_max = src_x[x];
            }
        }
        results[thread_id] = std::make_pair(thread_min, thread_max);
    }

    if(n_blocks > 1) {
        for(int i=1; i<n_blocks; ++i) {
            if(comp(results[i].first, results[0].first))
                results[0].first = results[i].first;
            if(comp(results[0].second, results[i].second))
                results[0].second = results[i].second;
        }
        return results[0];
    } else {
        return results[0];
    }
#else
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
#endif
}

template<typename T>
std::pair<T, T> minmax_element_image(const cv::Mat_<T>& src) {
    return minmax_element_image(src, std::less<T>());
}

} // namespace cvutil
