// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "contours_common.hpp"
#include <map>
#include <limits>
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/check.hpp"

using namespace std;
using namespace cv;

void cv::contourTreeToResults(CTree& tree,
                              int res_type,
                              OutputArrayOfArrays& _contours,
                              OutputArray& _hierarchy)
{
    // check if there are no results
    if (tree.isEmpty() || (tree.elem(0).body.isEmpty() && (tree.elem(0).first_child == -1)))
    {
        _contours.clear();
        return;
    }

    CV_Assert(tree.size() < (size_t)numeric_limits<int>::max());
    // mapping for indexes (original -> resulting)
    // -1 - based indexing
    vector<int> index_mapping(tree.size() + 1, -1);

    const int total = (int)tree.size() - 1;
    _contours.create(total, 1, 0, -1, true);
    {
        int i = 0;
        CIterator it(tree);
        while (!it.isDone())
        {
            const CNode& elem = it.getNext_s();
            CV_Assert(elem.self() != -1);
            if (elem.self() == 0)
                continue;
            index_mapping.at(elem.self() + 1) = i;
            CV_Assert(elem.body.size() < (size_t)numeric_limits<int>::max());
            const int sz = (int)elem.body.size();
            _contours.create(sz, 1, res_type, i, true);
            if (sz > 0)
            {
                Mat cmat = _contours.getMat(i);
                CV_Assert(cmat.isContinuous());
                elem.body.copyTo(cmat.data);
            }
            ++i;
        }
    }

    if (_hierarchy.needed())
    {
        _hierarchy.create(1, total, CV_32SC4, -1, true);
        Mat h_mat = _hierarchy.getMat();
        int i = 0;
        CIterator it(tree);
        while (!it.isDone())
        {
            const CNode& elem = it.getNext_s();
            if (elem.self() == 0)
                continue;
            Vec4i& h_vec = h_mat.at<Vec4i>(i);
            h_vec = Vec4i(index_mapping.at(elem.next + 1),
                          index_mapping.at(elem.prev + 1),
                          index_mapping.at(elem.first_child + 1),
                          index_mapping.at(elem.parent + 1));
            ++i;
        }
    }
}

void cv::copyBinarizedWithBorder(const Mat& src, Mat& dst, uchar fg)
{
    CV_Assert(src.type() == CV_8UC1 && !src.empty());
    const int width = src.cols, height = src.rows;
    dst.create(height + 2, width + 2, CV_8UC1);
    CV_Assert(dst.isContinuous());

    memset(dst.ptr(0), 0, (size_t)width + 2);
    memset(dst.ptr(height + 1), 0, (size_t)width + 2);

#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_uint8>::vlanes();
    const v_uint8 v_zero = vx_setzero_u8();
    const v_uint8 v_fg = vx_setall_u8(fg);
#endif
    for (int y = 0; y < height; y++)
    {
        const uchar* sptr = src.ptr(y);
        uchar* dptr = dst.ptr(y + 1);
        dptr[0] = 0;
        dptr[width + 1] = 0;
        dptr++;
        int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        for (; x <= width - vlanes; x += vlanes)
            v_store(dptr + x, v_and(v_ne(vx_load(sptr + x), v_zero), v_fg));
#endif
        for (; x < width; x++)
            dptr[x] = sptr[x] ? fg : (uchar)0;
    }
}
