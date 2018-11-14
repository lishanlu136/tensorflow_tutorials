#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2018/11/8 14:10
@Author     : Li Shanlu
@File       : lesson_5.py
@Software   : PyCharm
@Description: 关于训练相关的
"""

import tensorflow as tf
import os
"""
tf.gradients(y, [xs])计算y关于xs的梯度
"""
x = tf.Variable(2.0, name='x')
y = 2.0 * (x ** 3)
z = 3.0 + y ** 2
grad_z = tf.gradients(z, [x, y])
with tf.Session() as sess:
    sess.run(x.initializer)
    print(sess.run(grad_z))   # >> [768.0, 32.0]


"""
tf.train.Saver，用于保存图中的变量值
saves graph's variables in binary files
tf.train.Saver.save(sess, save_path=, global_step=None, latest_filename=None, meta_graph_suffix="meta", write_meta_graph=True, write_state=True)
"""
# Save parameters after 1000 steps
# define model
model = ......
loss = ......
# create a saver object
saver = tf.train.Saver()
# define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=loss,global_step=model.global_step)
# define max train step
max_steps = 10000
# launch a session to compute the graph
with tf.Session() as sess:
    # actual training loop
    for step in range(max_steps):
        sess.run([optimizer])
        if (step + 1) % 1000 == 0:
            saver.save(sess, 'checkpoint_directory/model_name', global_step=model.global_step)

# Restore the latest checkpoint
ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.dirname('checkpoints/checkpoint'))
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)


"""
tf.summary，可以将我们的训练过程保存在日志中，然后通过tensorboard查看训练情况
Visualize our summary statistics during our training
tf.summary.scalar
tf.summary.histogram
tf.summary.image
"""
###### step 1: create summaries
with tf.name_scope("summaries"):
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram("histogram_loss", loss)
    # merge them all
    summary_op = tf.summary.merge_all()  #像其他tf操作一样，summary也是一个操作，需要在sess中run

##### step 2: run them
loss_bath, _, summary = sess.run([model_loss, model.optimizer, model.summary_op], feed_dict=feed_dict)

##### step 3: write summaries to file
writer.add_summary(summary, global_step=step)
