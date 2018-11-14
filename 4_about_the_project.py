#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2018/11/7 17:13
@Author     : Li Shanlu
@File       : lesson_4.py
@Software   : PyCharm
@Description: 关于工程方面
"""
import tensorflow as tf

"""
Placeholders
TF编程分为两个阶段
1，设计你的图
2，在session中运行图中的操作
注意：在设计图的时候，你是不知道各个tensor的值的
如： f(x,y)=x*2+y ,我们可以将x，y用placeholder代替输入的实际值
tf.placeholder(dtype=, shape=None, name=None)
"""
# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3], name='a')
# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32, name='b')
# use the placeholder as you would a constant or a variable
c = a + b                               # Short for tf.add(a, b)
with tf.Session() as sess:
    # feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
    # fetch value of c
    print(sess.run(c, {a: [1, 2, 3]}))  # the tensor a is the key, not the string ‘a’
    # >> [6, 7, 8]

"""
如果想每次feed不一样的值,可以通过列表循环feed
with tf.Session() as sess:
    for a_value in list_of_values_for_a:
        print(sess.run(c, {a: a_value}))
"""


"""
Feeding values to TF ops
"""
# create operations, tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.multiply(a, 3)
with tf.Session() as sess:
    # define a dictionary that says to replace the value of 'a' with 15
    replace_dict = {a: 15}
    # Run the session, passing in 'replace_dict' as the value to 'feed_dict'
    print(sess.run(b, feed_dict=replace_dict))  # returns 45


"""
Avoid lazy loading
Separate the assembling of graph and executing ops
"""
# Normal loading:
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)              # you create the node for add node before executing the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/lesson_4', sess.graph)
    for _ in range(10):
        sess.run(z)           # z 在tensorboard中可以看到这个节点，并且在图的定义中只看到z只定义了一次
    print(tf.get_default_graph().as_graph_def())
    writer.close()

# Lazy loading:
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/lesson_4', sess.graph)
    for _ in range(10):
        sess.run(tf.add(x, y))  # 在tensorboard只能看到x，y两个节点，且在图的定义中能看十个add操作的节点，应该避免这种编程方式
    print(tf.get_default_graph().as_graph_def())
    writer.close()


"""
Name scope and variable scope
Group nodes together
with tf.name_scope(name):
with tf.variable_scope(name):
"""