#!-*- coding:utf-8 -*-
import tensorflow as tf


#input_x 的shape是(1, 2, 3, 3)
#input_test 的shape是(1, 3, 3, 2)
#通过这两个常量的定义，和具体卷积操作，可以理解data_format="NHWC"的作用
# [batch, in_height, in_width, in_channels]

input_x = tf.constant([[[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]],
                       [[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]
                      ]], dtype=tf.float32, name="Const-inputx")

input_test = tf.constant([[[[-1.31662965,0.28089154],
                            [-1.11473131,0.28089154],
                            [0.54786831, 0.28089154]],

                            [[-1.31662965,0.28089154],
                            [-1.11473131,0.28089154],
                            [0.54786831, 0.28089154]],

                            [[-1.31662965,0.28089154],
                            [-1.11473131,0.28089154],
                            [0.54786831, 0.28089154]]]], dtype=tf.float32)

# [filter_height, filter_width, in_channels, out_channels]
#卷积核，在神经网络训练的过程中，卷积核的值一般作为训练的权重
filter = tf.Variable(tf.random_normal([2, 2, 2, 3]))

#test_va = tf.Variable(tf.random_normal([1, 3, 3, 2]))
#
#Must have strides[0] = strides[3] = 1. For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].
#详细介绍：查看README.md
conv_output1 = tf.nn.conv2d(input_test, filter, [1, 2, 2, 1], padding='SAME', data_format="NHWC")
conv_output2 = tf.nn.conv2d(input_x, filter, [1, 1, 1, 1], padding='SAME', data_format="NCHW")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print input_x.shape
    print input_test.shape
    print filter.eval()
    #print test_va.eval()
    print conv_output1.eval()
    print conv_output2.eval()