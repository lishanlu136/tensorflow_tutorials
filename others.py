#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2018/11/8 16:03
@Author     : Li Shanlu
@File       : lesson_7.py
@Software   : PyCharm
@Description:
"""
import tensorflow as tf
tf.nn.conv2d(input=, filter=, strides=, padding=, use_cudnn_on_gpu=True, data_format="NHWC", name=None)

"""
Input: Batch size x Height x Width x Channels
Filter: kernel_Height x kernel_Width x Input Channels x Output Channels  (e.g. [5, 5, 3, 64])
Strides: 4 element 1-D tensor, strides in each direction  (often [1, 1, 1, 1] or [1, 2, 2, 1])
Padding: ‘SAME’ or ‘VALID’
Data_format: default to NHWC
"""