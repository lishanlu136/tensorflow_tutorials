#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2018/11/13 11:44
@Author     : Li Shanlu
@File       : net_example_for_define_vars.py
@Software   : PyCharm
@Description: 用一个网络来实践怎么定义重复利用的变量
"""
import tensorflow as tf


def conv(input, kernel_shape, bias_shape):
    weights = tf.get_variable(name='weights', shape=kernel_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable(name='biases', shape=bias_shape, initializer=tf.constant_initializer())
    conv_result = tf.nn.conv2d(input, weights, [1,2,2,1], padding='SAME')
    return tf.nn.relu(tf.add(conv_result, biases))


def net(input):
    with tf.variable_scope('net1'):
        conv1_out = conv(input, [5, 5, 1, 32], [32])
    with tf.variable_scope('net2'):
        conv2_out = conv(conv1_out, [5, 5, 32, 64], [64])
    return None


if __name__ == '__main__':
    input = tf.placeholder(tf.float32, shape=[1, 28, 28, 1])
    with tf.variable_scope('test') as scope:
        net(input)
        scope.reuse_variables()
        net(input)
    for var in tf.trainable_variables():
        print(var)

"""
>>
<tf.Variable 'test/net1/weights:0' shape=(5, 5, 1, 32) dtype=float32_ref>
<tf.Variable 'test/net1/biases:0' shape=(32,) dtype=float32_ref>
<tf.Variable 'test/net2/weights:0' shape=(5, 5, 32, 64) dtype=float32_ref>
<tf.Variable 'test/net2/biases:0' shape=(64,) dtype=float32_ref>
____________________________________________________________________________
总结：
在net第一次调用时定义了四个变量，通过使用函数reuse_variables()，在第二次调用net的时候，共享了上一次的变量参数。
"""