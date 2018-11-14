#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2018/11/13 9:18
@Author     : Li Shanlu
@File       : Variable_and_get_variable.py
@Software   : PyCharm
@Description: 解释并验证tensorflow中placeholder、Variable、get_variable的区别
"""
import tensorflow as tf
print("example_1 start ......")
v1 = tf.placeholder(tf.float32, shape=[1])
print(v1.name)
v1 = tf.placeholder(tf.float32, shape=[1], name='variable')
print(v1.name)
v1 = tf.placeholder(tf.float32, shape=[1], name='variable')
print(v1.name)

print(type(v1))
var_set = tf.trainable_variables()
print(len(var_set))
print("example_1 end ......\n")
"""
>> 
Placeholder:0
variable:0
variable_1:0
<class 'tensorflow.python.framework.ops.Tensor'>
0
___________________________________________________________________________
总结:
1.如果没有给placeholder指定名字，那么默认是Placeholder、Placeholder_1。 
2.声明重复名字的placeholder是允许的，但是系统会按照顺序进行编号name、name_1。 
3.placeholder是Tensor类型。 
4.调用存在可训练变量，长度为0，所以placeholder属于不可训练参数。
"""

print("example_2 start ......")
v2 = tf.Variable([1], dtype=tf.float32)
print(v2.name)
v2 = tf.Variable([1], dtype=tf.float32, name='var2')
print(v2.name)
v2 = tf.Variable([1], dtype=tf.float32, name='var2')
print(v2.name)

print(type(v2))
vars = tf.trainable_variables()
for i in vars:
    print(i)
print("example_2 end ......\n")

"""
>> 
Variable:0
var2:0
var2_1:0
<class 'tensorflow.python.ops.variables.Variable'>
<tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'var2:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'var2_1:0' shape=(1,) dtype=float32_ref>
_________________________________________________________________________
总结：
1.如果没有给variable指定名字，那么默认是Variable、Variable_1。 
2.声明重复名字的variable是允许的，但是系统会按照顺序进行编号name、name_1。 
3.variable是Variable类型。 
4.variable是可训练的。 
5.variable创建并赋值的时候，会构建一个对象存储在内存中。
"""

print("example_3 start ......")
"""
v3 = tf.get_variable(shape=[1], name='get_var3')
print(v3.name)
v3 = tf.get_variable(shape=[1], name='get_var3')
print(v3.name)
"""
# 结果会报错： ValueError: Variable get_var3 already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
with tf.variable_scope('test_reuse') as scope:
    v4 = tf.get_variable(shape=[1], name='get_var4')
    print(v4.name)
    scope.reuse_variables()
    v4 = tf.get_variable(shape=[1], name='get_var4')
    print(v4.name)
with tf.variable_scope('test') as scope:
    v5 = tf.get_variable(shape=[1], name='get_var5')
    print(v5.name)
with tf.variable_scope('test1') as scope:
    v5 = tf.get_variable(shape=[1], name='get_var5')
    print(v5.name)
new_vars = tf.trainable_variables()
for i in new_vars:
    print(i)
print("example_3 end ......")

"""
>> 
test_reuse/get_var4:0
test_reuse/get_var4:0
test/get_var5:0
test1/get_var5:0
<tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'var2:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'var2_1:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'test_reuse/get_var4:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'test/get_var5:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'test1/get_var5:0' shape=(1,) dtype=float32_ref>
_________________________________________________________________________________
总结：
1.用get_variable不可以重复创建相同名字的变量。 
2.使用reuse_variable()可以实现变量重用，但是要放在一个变量域中variable_scope，重用之后还是只有一个可训练变量。 
3.可以将两个重名的变量分别放在两个variable_scope中用get_variable定义。也就是说变量是可以被域限定的,这时就是两个可训练变量了。
"""
