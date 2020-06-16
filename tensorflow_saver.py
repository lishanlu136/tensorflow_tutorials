#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/4/22 17:16
@Author     : Li Shanlu
@File       : tensorflow_saver.py
@Software   : PyCharm
@Description: 描述tensorflow的saver相关的东西
"""

import tensorflow as tf
# 经常用到的方法
sess = tf.Session()
saver = tf.train.Saver()
saver.save(sess,
           save_path,
           global_step=None,
           latest_filename=None,
           meta_graph_suffix="meta",
           write_meta_graph=True,
           write_state=True)                      # 用于保存训练的模型

saver.recover_last_checkpoints(checkpoint_paths)  # 用于从最近一次的训练结果恢复模型

saver.restore(sess, save_path)                    # 加载模型，可以指定加载某个模型，不一定非得最近一次


# -----------------------------------------------------------------------------------------------------------------
# 微调网络过程中遇到的问题
# 1.网络只需加载一部分预训练模型的权重怎么办；或者说网络中某些层的权重，预训练模型中没有。
# 解决办法：在定义saver对象的时候，把网络中这些层排除掉即可。然后在用restore从预训练模型中加载权重时就不会报错了，
# 网络其余层没有从预训练模型加载权重的就需要初始化啦。
"""
# 比如我网络中Logits层在预训练模型中没有
# 指定加载某些变量的权重
all_vars = tf.trainable_variables()
var_to_skip = [v for v in all_vars if v.name.startswith('Logits')]
print("got pretrained model, var_to_skip:\n" + " \n".join([x.name for x in var_to_skip]))
var_to_restore = [v for v in all_vars if not (v.name.startswith('Logits'))]
saver = tf.train.Saver(var_to_restore, max_to_keep=20)
sess.run(tf.global_variables_initializer())  # 初始化其余层的变量
saver.restore(sess, pretrained_model)  # 利用saver.restore恢复指定层的权重
"""

# 2.保存是时候还要用之前定义的saver吗？
# 我们之前定义的saver为了正确加载预训练模型，是把网络中以‘Logits’开头的变量排除了的；
# 所以，如果还用这个saver来save训练模型的话，模型中会没有‘Logits’层的权重的。
# 解决办法：重新再定义一个包含网络全部变量的saver对象用于保存模型，一个图中可以定义多个saver对象哟。

