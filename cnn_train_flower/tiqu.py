# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 12:07:59 2017

@author: Administrator
"""

import tensorflow as tf
from skimage import io,transform
import numpy as np
from PIL import Image


#-----------------构建网络----------------------
#占位符
x=tf.placeholder(tf.float32,shape=[None,100,100,3],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

#第一个卷积层（100——>50)
conv1=tf.layers.conv2d(
      inputs=x,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

#第二个卷积层(50->25)
conv2=tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

#第三个卷积层(25->12)
conv3=tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

#第四个卷积层(12->6)
conv4=tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

#全连接层
dense1 = tf.layers.dense(inputs=re1,
                      units=1024,
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.nn.l2_loss)
dense2= tf.layers.dense(inputs=dense1,
                      units=512,
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.nn.l2_loss)
logits= tf.layers.dense(inputs=dense2,
                        units=5,
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.nn.l2_loss)

#---------------------------网络结束---------------------------
#%%
#取出所有参与训练的参数
params=tf.trainable_variables()
print("Trainable variables:------------------------")

#循环列出参数
for idx, v in enumerate(params):
     print("  param {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

#%%
#读取图片
img=io.imread('F:\\py3workspace\\cnn_train_animal\\cat.jpg')
#resize成100*100
img=transform.resize(img,(100,100))
#三维变四维（100，100，3）-->(1,100,100,3)
img=img[np.newaxis,:,:,:]
img=np.asarray(img,np.float32)
sess=tf.Session()
sess.run(tf.global_variables_initializer())

checkpoint_dir = '.\\' #表示checkpoints文件的保存路径，例子中使用当前路径F:\\py3workspace\\train_captcha\\
saver = tf.train.Saver()
# 检查是否有checkpoint
checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
if checkpoint and checkpoint.model_checkpoint_path:
    #saver.restore(sess, checkpoint.model_checkpoint_path)
    #print(checkpoint.model_checkpoint_path.rsplit('-',1)[1])
    start_step = int(checkpoint.model_checkpoint_path.rsplit('-',1)[1])
    print(start_step)
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))


#提取最后一个全连接层的参数 W和b
W=sess.run(params[12])
b=sess.run(params[13])
print(W)
#提取第二个全连接层的输出值作为特征
fea=sess.run(dense2,feed_dict={x:img})
print(fea)


# 卷积
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# 源码的位置在tensorflow/python/ops下nn_impl.py和nn_ops.py
# 这个函数接收两个参数,x 是图像的像素, w 是卷积核
# x 张量的维度[batch, height, width, channels]
# w 卷积核的维度[height, width, channels, channels_multiplier]
# tf.nn.conv2d()是一个二维卷积函数,
# stirdes 是卷积核移动的步长,4个1表示,在x张量维度的四个参数上移动步长
# padding 参数'SAME',表示对原始输入像素进行填充,卷积后映射的2D图像与原图大小相等
# 填充,是指在原图像素值矩阵周围填充0像素点
# 如果不进行填充,假设 原图为 32x32 的图像,卷积和大小为 5x5 ,卷积后映射图像大小 为 28x28
