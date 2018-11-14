#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2018/11/7 11:50
@Author     : Li Shanlu
@File       : lesson_2.py
@Software   : PyCharm
@Description: 用tensorboard可视化
"""
import tensorflow as tf
"""
Visualize graph with tensorboard
Go to terminal, run:
$ python [yourprogram].py
$ tensorboard --logdir="./graphs" --port 6006
Then open your browser and go to: http://localhost:6006/
"""
"""
# add ops to the default graph
a = tf.constant(2, name='a')  # 明确节点名字为a
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')  # 明确节点名字为add
with tf.Session() as sess:
    # add this line to use tensorboard.
    writer = tf.summary.FileWriter('./graphs/lesson_2', graph=sess.graph)
    print('x = ', sess.run(x))
writer.close() #close the writer when you're done using it.
"""



"""
More constants
tf.constant(value=, dtype=None, shape=None, name="Const", verify_shape=False)
"""
a = tf.constant(value=[2, 2], name='a')
b = tf.constant(value=[[0, 1], [2, 3]], name='b')
x = tf.add(a, b, name="add")
y = tf.multiply(a, b, name="multiply")
with tf.Session() as sess:
    x_, y_ = sess.run([x, y])
    print("x = ", x_)
    print("y = ", y_)
    print(sess.graph.as_graph_def())   # 打印图，可以看到常量a,b的值也存储在图的定义中，如果值很大的话，对加载图开销很大; 所以只对基元类型使用常量，为需要更多内存的数据使用变量或阅读器

