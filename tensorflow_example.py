#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2018/9/12 14:06
# @Author     : Li Shanlu
# @File       : tensorflow_example.py
# @Software   : PyCharm
# @Description: Show some examples for using tensorflow.

import tensorflow as tf

# Show how to define var
with tf.name_scope("example_test_name"):
    init = tf.constant_initializer(value=1)
    var_1 = tf.get_variable(name="name_var1", shape=[1], dtype=tf.float32, initializer=init)
    var_2 = tf.Variable(initial_value=[2], name="name_var2", dtype=tf.float32)
    var_21 = tf.Variable(initial_value=[2.1], name="name_var2", dtype=tf.float32)
    var_22 = tf.Variable(initial_value=[2.2], name="name_var2", dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(var_1.name)
    print(sess.run(var_1))
    print(var_2.name)
    print(sess.run(var_2))
    print(var_21.name)
    print(sess.run(var_21))
    print(var_22.name)
    print(sess.run(var_22))

v1 = tf.placeholder(tf.float32, shape=[1])
print(v1.name)
v1 = tf.placeholder(tf.float32, shape=[1], name='users')
print(v1.name)
v1 = tf.placeholder(tf.float32, shape=[1], name='users')
print(v1.name)
print(type(v1))

phs = tf.trainable_variables()
print(len(phs))
