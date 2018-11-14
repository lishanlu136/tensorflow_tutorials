#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2018/11/7 11:13
@Author     : Li Shanlu
@File       : lesson_1.py
@Software   : PyCharm
@Description:
"""
import tensorflow as tf
a = tf.add(3, 5)
"""
sess = tf.Session()
print("a =", sess.run(a))
sess.close()
"""
# 可以用下面的代码替换上面被注释的部分
with tf.Session() as sess:
    print("a = ", sess.run(a))

"""
More graphs
# 因为sess.run没有调用useless这个op，所有它不会被计算
"""
x = 2
y = 3
add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.Session() as sess:
    z = sess.run(pow_op)
    # z, not_useless = sess.run([pow_op, useless])

"""
Distributed computation
"""
# Creates a graph.
with tf.device('/gpu:2'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
    c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print("c = ", sess.run(c))

"""
怎么同时定义多个图
"""
g1 = tf.get_default_graph()
g2 = tf.Graph()
# add ops to the default graph
with g1.as_default():
    a = tf.constant(3)
    b = tf.constant(4)
    c = tf.add(a, b)
# add ops to the user created graph
with g2.as_default():
    d = tf.constant(5)
    e = tf.constant(6)
    f = tf.multiply(d, e)
with tf.Session(graph=g1) as sess1:  # session is run on the graph g1
    print("c = ", sess1.run(c))
with tf.Session(graph=g2) as sess2:  # session is run on the graph g2
    print("f = ", sess2.run(f))

