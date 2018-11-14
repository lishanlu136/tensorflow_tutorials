#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2018/9/11 10:48
# @Author     : Li Shanlu
# @File       : show_graph_in_tensorboard.py
# @Software   : PyCharm
# @Description: Read a net graph from .pb file and show it in tensorboard.
import tensorflow as tf

model = '/data1/lishanlu/download/tensorflow/model.pb'  # 请将这里的pb文件路径改为自己对应的pb文件
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('./log/model_graph/', graph)