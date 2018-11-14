#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2018/11/13 9:14
@Author     : Li Shanlu
@File       : name_scope_and_variable_scope.py
@Software   : PyCharm
@Description: 解释并验证name_scope和variable_scope的区别
"""
import tensorflow as tf

with tf.name_scope('n_s') as ns:
    v1 = tf.Variable([1], name='var1')
    v2 = tf.Variable([1], name='var1')
    v3 = tf.get_variable(shape=[1], name='var3')
    # v4 = tf.get_variable(shape=[1], name='var3')  # 重复定义var3，会出错，ValueError: Variable var3 already exists, disallowed.
    # ns.reuse_variables()   # 报错，AttributeError: 'str' object has no attribute 'reuse_variables'
    # v4 = tf.get_variable(shape=[1], name='var3')

    with tf.variable_scope('v_s') as vs:
        v5 = tf.Variable([1], name='var1')
        v6 = tf.Variable([1], name='var2')
        v7 = tf.Variable([1], name='var2')
        v8 = tf.get_variable(shape=[1], name='var3')
        # v9 = tf.get_variable(shape=[1], name='var3') # 重复定义var3，会出错，ValueError: Variable v_s/var3 already exists, disallowed.
        vs.reuse_variables()   # 这句必须加上才能重复利用上面这个变量，v8和v9其实是一个变量，占用同一块内存
        v9 = tf.get_variable(shape=[1], name='var3')

print('v1 name: ', v1.name)
print('v2 name: ', v2.name)
print('v3 name: ', v3.name)
# print('v4 name: ', v4.name)
print('v5 name: ', v5.name)
print('v6 name: ', v6.name)
print('v7 name: ', v7.name)
print('v8 name: ', v8.name)
print('v9 name: ', v9.name)
var_set = tf.trainable_variables()
for i in var_set:
    print(i)

"""
>>
v1 name:  n_s/var1:0
v2 name:  n_s/var1_1:0
v3 name:  var3:0
v5 name:  n_s/v_s/var1:0
v6 name:  n_s/v_s/var2:0
v7 name:  n_s/v_s/var2_1:0
v8 name:  v_s/var3:0
v9 name:  v_s/var3:0
<tf.Variable 'n_s/var1:0' shape=(1,) dtype=int32_ref>
<tf.Variable 'n_s/var1_1:0' shape=(1,) dtype=int32_ref>
<tf.Variable 'var3:0' shape=(1,) dtype=float32_ref>
<tf.Variable 'n_s/v_s/var1:0' shape=(1,) dtype=int32_ref>
<tf.Variable 'n_s/v_s/var2:0' shape=(1,) dtype=int32_ref>
<tf.Variable 'n_s/v_s/var2_1:0' shape=(1,) dtype=int32_ref>
<tf.Variable 'v_s/var3:0' shape=(1,) dtype=float32_ref>
_____________________________________________________________________
总结：
1.利用get_variable声明变量时，不受name_scope的影响
2.name_scope只影响用tf.Variable形式声明的变量名
3.不在同一个variable_scope的get_variable是互不影响的
4.在name_scope里面不能用get_variable来重复利用变量，想重复利用变量只能在variable_scope里面用get_variable形式声明变量
"""
