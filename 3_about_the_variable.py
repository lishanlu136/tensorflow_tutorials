#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2018/11/7 16:31
@Author     : Li Shanlu
@File       : lesson_3.py
@Software   : PyCharm
@Description: 关于变量
"""

import tensorflow as tf
"""
Tensors filled with a specific value
"""
"""
tf.zeros(shape=, dtype=tf.float32, name=None)   #和numpy.zeros相似
tf.zeros_like(tensor=, dtype=None, name=None, optimize=True)   #创建和所给的tensor一样大小，类型的tensor，但元素全为零，和numpy.zeros_like相似
tf.ones(shape=, dtype=tf.float32, name=None)
tf.ones_like(tensor=, dtype=None, name=None, optimize=True)
tf.fill(dims=, value=, name=None)  # creates a tensor filled with a scalar value. 如：tf.fill([2, 3], 8) ==> [[8, 8, 8], [8, 8, 8]]
tf.linspace(start=, stop=, num=, name=None) # tf.linspace(10.0, 13.0, 4) ==> [10.0 11.0 12.0 13.0]
tf.range(start=, limit=None, delta=1, dtype=None, name="range") # tf.range(3, 18, 3) ==> [3, 6, 9, 12, 15]

# Tensor objects are not iterable
for _ in tf.range(4): # TypeError

# Randomly Generated Constants
tf.random_normal(shape=, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape=, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.random_uniform(shape=, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
tf.random_shuffle(value=, seed=None, name=None)
tf.random_crop(value=, size=, seed=None, name=None)
tf.multinomial(logits=, num_samples=, seed=None, name=None)
tf.random_gamma(shape=, alpha=, beta=None, dtype=tf.float32, seed=None, name=None)
tf.set_random_seed(seed=)
"""


"""
创建变量
"""
# create variable a with scalar value
a = tf.Variable(2, name="scalar_a")
# create variable b as a vector
b = tf.Variable([2, 3], name="vector_b")
# create variable c as a 2x2 matrix
c = tf.Variable([[0, 1], [2, 3]], name="matrix_c")
# create variable W as 784 x 10 tensor, filled with zeros
W = tf.Variable(tf.zeros([784, 10]), name="W")
"""
为什么是tf.Variable,而不是tf.variable呢？ 因为定义常量就是tf.constant呀
答：因为tf.Variable是一个类，而tf.constant是一个操作(op)
tf.Variable包含有以下方法：
x = tf.Variable(...) 
x.initializer # init op
x.value() # read op
x.assign(...) # write op
x.assign_add(...) # and more
"""


"""
使用变量之前必须初始化
"""
# The easiest way is initializing all variables at once:
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
# Initialize only a subset of variables:
init_ab = tf.variables_initializer([a, b], name="init_ab")
with tf.Session() as sess:
    sess.run(init_ab)
# Initialize a single variable
W = tf.Variable(tf.zeros([784, 10]), name="W")
with tf.Session() as sess:
    sess.run(W.initializer)

# W is a random 700 x 100 variable object
W = tf.Variable(tf.truncated_normal([700, 10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W)          # 打印W这个tensor, >> <tf.Variable 'Variable:0' shape=(700, 10) dtype=float32_ref>
    print(W.eval())   # 打印W的值, [[....],[...],..]
    
"""
tf.Variable.assign()
"""
w = tf.Variable(10)
assign_op = w.assign(100)      # tf.Variable.assign()也是一个op，也是需要sess.run()才会起作用的
with tf.Session() as sess:
    # sess.run(w.initializer)  # 可以不必初始化变量w了，因为assign_op自带初始化功能
    sess.run(assign_op)
    print("w = ", w.eval())    # >> 100

"""
assign_add() and assign_sub()
这两个函数就需要初始化原来的变量了，因为它们需要调用变量原来的值
"""
my_var = tf.Variable(10)
with tf.Session() as sess:
    sess.run(my_var.initializer)
    # increment by 10
    sess.run(my_var.assign_add(10))  # >> 20
    # decrement by 2
    sess.run(my_var.assign_sub(2))   # >> 18

"""
每个session下维护自己的变量副本
"""
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10)))   # >> 20
print(sess2.run(W.assign_sub(2)))    # >> 8， 而不是22
print(sess1.run(W.assign_add(100)))  # >> 120
print(sess2.run(W.assign_sub(50)))   # >> -42
sess1.close()
sess2.close()

"""
用一个变量初始化另一个变量
"""
# W is a random 700 x 100 tensor
W = tf.Variable(tf.truncated_normal([700, 10]))
# U = tf.Variable(2 * W)                # 不是很安全
U = tf.Variable(2 * W.initial_value)    # 安全，在使用W之前，确保W被初始化了
