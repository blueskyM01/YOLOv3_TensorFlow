# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
    '''
    功能：对网络的输出做“非极大值（nms）”抑制处理,
    :param boxes: 网络预测输出的box（已经反解到原图？）, 每个点输出3个box, 也就是每个点预测3个目标
                  tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
    :param scores: 网络输出的分数： score=置信度 * 概率， 每个点输出3个score , 也就是每个点预测3个目标
                   tensor of shape [1, 10647, num_classes], score=conf*prob
    :param num_classes: integer, maximum number of predicted boxes you'd like, default is 50. coco数据集为：80
    :param max_boxes: keep at most nms_topk outputs after nms, 150
    :param score_thresh: score = pred_confs * pred_probs. set lower for higher recall, 0.01
    :param nms_thresh: 0.45
    :return: boxes: 网络输出经过过滤后的最终目标矩形框， shape=(-1,4)
             score: 网络输出经过过滤后的最终目标分数， shape=(-1,）
             label: 网络输出经过过滤后的最终目标标签， shape=(-1,）
    '''
    """
    Perform NMS on GPU using TensorFlow.

    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        nms_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
    score = tf.reshape(scores, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    '''
    tf.greater_equal(x, y): 若x >= y, x对应的元素返回True，反之False
    例：
    a = tf.constant([[1,2,3],
                     [4,5,6]])
    b = tf.constant(3)
    c = tf.greater_equal(a,b):
                              [[False False  True]
                               [ True  True  True]]
    mask： 将大于阈值的目标打上True的标， 小于阈值的打上False标签，得到一个与score相同的掩膜
    '''
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        '''
        tf.boolean_mask(tensor, mask): 返回tensor与mask中True元素同下标的部分
        例：
        a = tf.constant([[1,2,3],
                         [4,5,6],
                         [7,8,9]])
        b = tf.constant([False,True, True])
        c = tf.boolean_mask(a,b):
                                 [[4 5 6]
                                  [7 8 9]]
        filter_boxes: 将分数大于阈值的目标矩形框挑出来, shape=(-1, 4)
        filter_score: 将分数大于阈值的属于第i类目标的分数取出来, shape=(-1,)
        '''
        filter_boxes = tf.boolean_mask(boxes, mask[:,i])
        filter_score = tf.boolean_mask(score[:,i], mask[:,i])

        '''
        tf.image.non_max_suppression（boxes, scores, max_output_size, iou_threshold）: 返回过滤后剩下的filter_boxes的下标
        nms_indices: shape=(-1,)
        '''
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=nms_thresh, name='nms_indices')
        '''
        
        '''
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0) # shape=(-1,4)
    score = tf.concat(score_list, axis=0) # shape=(-1,)
    label = tf.concat(label_list, axis=0) # shape=(-1,)
    return boxes, score, label


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]


def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    Perform NMS on CPU.
    Arguments:
        boxes: shape [1, 10647, 4]
        scores: shape [1, 10647, num_classes]
    """

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:,i][indices]
        if len(filter_boxes) == 0: 
            continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0: 
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label