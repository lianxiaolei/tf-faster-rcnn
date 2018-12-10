# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.cython_bbox import bbox_overlaps
from model.bbox_transform import bbox_transform


def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    """
    Same as the anchor target layer in original Fast/er RCNN
    :param gt_boxes [None, 5]
    """

    A = num_anchors  # 9
    total_anchors = all_anchors.shape[0]  # 1764
    K = total_anchors / num_anchors  # 1764/9

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # map of shape (..., H/16, W/16)
    height, width = rpn_cls_score.shape[1:3]

    # only keep anchors inside the image
    # inds_inside 是没有越界的框在ndarray中的索引号
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]

    # keep only inside anchors
    # anchors变为只完全在图像内的anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        # ascontiguousarray 返回一个地址连续的数组
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    # overlaps[N, K] N个anchors与K个真实box的交并比

    # 求与每个anchors交并比最大的真实box的index k
    argmax_overlaps = overlaps.argmax(axis=1)
    # 用坐标求出来具体的值
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

    # 求与每个真实box交并比最大的anchor的index n
    gt_argmax_overlaps = overlaps.argmax(axis=0)  # shape(1, K)
    # 用坐标求出来具体的值 shape(1, K), 取出来抓到的每列的最大值
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]

    # 返回被选中的元素的x和y两个list
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives
        # 将交并比小于阈值的设为背景，即label为0
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    # 将与每个gt_box交并比最大的 anchor的label设为1
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    # 将与每个anchors交并比最大的真实box所在的anchors index的label设为1
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0  # ???

    # subsample positive labels if we have too many
    # 对正例重采样
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

    # np.where tuple类型，第一维的index列表
    # fg_inds label=1的第一维坐标
    fg_inds = np.where(labels == 1)[0]

    # 如果检测出的正例个数大于阈值
    if len(fg_inds) > num_fg:
        # 随机选出过多的index将其label设为-1
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    # 对负例重采样 labels现在全为-1，长度是所有的图像内anchor的个数
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)

    # 统计所有在图像内的anchor的BBR value
    # shape(num anchors in image, 4)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    # initialize inside weight
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)  # ???什么狗

    # initialize outside weight same as initialization of inside weight
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)

    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:  # ???什么辣鸡
        # uniform weighting of examples (given non-uniform sampling)
        # 前背景数
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            np.sum(labels == 0))

    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    # total anchor 1764
    # labels num anchor in image
    # shape(1764, num_inside)
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    # height, width, A = H/16, W/16, 9
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)  # transpose to 1, 9, H/16, W/16
    labels = labels.reshape((1, 1, A * height, width))  # reshape to 1, 1, 9 * H/16, W/16

    # shape(1, 1, 9 * H/16, W/16)
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights
    # shape(1,1,9*H/16,W/16), shape(1,H/16,W/16,9*4), shape(1, height, width, A * 4), shape(1, height, width, A * 4)
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """
     Unmap a subset of item (data) back to the original set of items (of size count)
    :param data: labels shape(1, inside inds)
    :param count: total anchor 1764
    :param inds: inside inds
    :param fill: -1
    :return:
    """
    # 如果是一维
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        # shape(1764, num_inside), 两tuple相加为拼接成一个tuple
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """
    Compute bounding-box regression targets for an image.
    :param ex_rois: 所有在图像内的anchor
    :param gt_rois: 所有在图像内的gt_box
    :return: shape(num anchors in image, 4)
    """

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
