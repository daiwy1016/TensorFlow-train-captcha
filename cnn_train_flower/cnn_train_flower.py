# -*- coding: utf-8 -*-

from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

path='F:/py3workspace/cnn_train_animal/flower_photos/'

#数据：http://download.tensorflow.org/example_images/flower_photos.tgz
#花总共有五类，分别放在5个文件夹下。


#将所有的图片resize成100*100
w=100
h=100
c=3


#读取图片
def read_img(path):
    #是一个典型的列表生成式，左边是列表元素（X），右边是条件，说明列表的元素都是路径。
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    #enumerate在字典上是枚举、列举的意思,是python的内置函数
    #对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
    #idx:索引，folder：是值
    for idx,folder in enumerate(cate):
        #返回所有匹配的文件路径列表
        #获取指定目录下的所有图片
        for im in glob.glob(folder+'/*.jpg'):
            #print('reading the images:%s'%(im))
            #读取图片
            img=io.imread(im)
            ##resize成w*h
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
        print(u"读取图片完毕：目录为",folder)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
    #asarray将输入数据（列表的列表，元组的元组，元组的列表等）转换为矩阵形式
    #https://www.2cto.com/kf/201406/311743.html 详情请看
data,label=read_img(path)


#打乱顺序
#shape[0]就是读取矩阵第一维度的长度 也就是读取多少张图片
num_example=data.shape[0]
#arange([start,] stop[, step,], dtype=None)根据start与stop指定的范围以及step设定的步长，生成一个 ndarray。 dtype : dtype
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


#将所有数据分为训练集和验证集
ratio=0.8
s=np.int(num_example*ratio)
#data[:s] 到0-s的数组
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

#-----------------构建网络----------------------
#占位符
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
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
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2= tf.layers.dense(inputs=dense1,
                      units=512,
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits= tf.layers.dense(inputs=dense2,
                        units=5,
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
#---------------------------网络结束---------------------------

loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


#训练和测试数据，可将n_epoch设置更大一些

n_epoch=10
batch_size=64
checkpoint_dir = '.\\' #表示checkpoints文件的保存路径，例子中使用当前路径F:\\py3workspace\\train_captcha\\
# 问题： tf.Session()和tf.InteractiveSession()的区别？
# 答案：
# 唯一的区别在于：tf.InteractiveSession()加载它自身作为默认构建的session，tensor.eval()和operation.run()取决于默认的session.
# 换句话说：InteractiveSession 输入的代码少，原因就是它允许变量不需要使用session就可以产生结构。
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1)
for epoch in range(n_epoch):
    start_time = time.time()

    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
        print("*****第%d次********train loss: %f,train acc: %f" %(n_batch,(train_loss/ n_batch), (train_acc/ n_batch)))
        if n_batch%10 == 0:
            saver.save(sess, checkpoint_dir +"crack_flower.model", global_step=n_batch)

    #print("   train loss: %f" % (train_loss/ n_batch))
    #print("   train acc: %f" % (train_acc/ n_batch))

    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
        print("*****第%d次*********validation loss: %f,validation acc: %f" % (n_batch,(val_loss/ n_batch),(val_acc/ n_batch)))

    #print("   validation loss: %f" % (val_loss/ n_batch))
    #print("   validation acc: %f" % (val_acc/ n_batch))

sess.close()
