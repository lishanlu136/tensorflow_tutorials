#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2018/11/8 16:20
@Author     : Li Shanlu
@File       : lesson_9.py
@Software   : PyCharm
@Description: 关于随机的控制，和文件的队列读取方式
"""
import tensorflow as tf
import numpy as np

"""
Control Randomization
"""
# Op level random seed
# sessions keep track of random state, each new session restarts the random state.
# each op keeps its own seed.
my_var = tf.Variable(tf.truncated_normal((-1.0, 1.0), stddev=0.1, seed=666))

# Graph level seed
tf.set_random_seed(seed=666)


"""
Data Readers
Different Readers for different file types

tf.TextLineReader
Outputs the lines of a file delimited by newlines
E.g. text files, CSV files

tf.FixedLengthRecordReader
Outputs the entire file when all files have same fixed lengths
E.g. each MNIST file has 28 x 28 pixels, CIFAR-10 32 x 32 x 3

tf.WholeFileReader
Outputs the entire file content

tf.TFRecordReader
Reads samples from TensorFlow’s own binary format (TFRecord)

tf.ReaderBase
To allow you to create your own readers
"""


"""
Read in files from queues
"""
filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)


"""
利用队列读取data和label
"""
N_SAMPLES = 10
NUM_THREADS = 4
all_data = 10 * np.random.randn(N_SAMPLES, 4) + 1
all_target = np.random.randint(0, 2, size=N_SAMPLES)
print("all data:", all_data)
print("all target:", all_target)

queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])
enqueue_op = queue.enqueue_many([all_data, all_target])  # A common practice is to enqueue all data at once, but dequeue one by one
data_sample, label_sample = queue.dequeue()

qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)
with tf.Session() as sess:
    # create a coordinator, launch the queue runner threader.
    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    for step in range(50):  # do to 50 iterations
        if coord.should_stop():
            break
        one_data, one_label = sess.run([data_sample, label_sample])
        print(one_data, one_label)
    coord.request_stop()
    coord.join(enqueue_threads)