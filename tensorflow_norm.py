#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/4/22 16:17
@Author     : Li Shanlu
@File       : tensorflow_norm.py
@Software   : PyCharm
@Description: 描述tf.nn.l2_normalize的使用和tf.norm方法的异同
"""

import tensorflow as tf

input_data = tf.constant([[1.0,2,3],[2.0,3,4],[3.0,4,5]])
output_1 = tf.nn.l2_normalize(input_data, dim=1, epsilon=1e-10, name='nn_l2_norm')
normal = tf.norm(input_data, axis=1, keep_dims=True, name='normal')  # 求每行对应的L2范数(欧式距离)
output_2 = tf.div(input_data, normal, name='div_normal')
normal_1 = tf.norm(output_1, axis=1, keep_dims=True, name='normal_1')
output_3 = tf.div(output_1, normal_1, name='div_normal_1')

with tf.Session() as sess:
    print("input_data:\n", sess.run(input_data))
    print("output_1:\n", sess.run(output_1))
    print("normal:\n", sess.run(normal))
    print("output_2:\n", sess.run(output_2))
    print("normal_1:\n", sess.run(normal_1))
    print("output_3:\n", sess.run(output_3))


