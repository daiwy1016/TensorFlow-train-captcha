#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-09 17:25:02
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$


from gen_captcha import gen_captcha_text_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET

import numpy as np
import tensorflow as tf

"""
text, image = gen_captcha_text_and_image()
print  ("验证码图像channel:", image.shape)  # (60, 160, 3)
# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = len(text)
print   ("验证码文本最长字符数", MAX_CAPTCHA)  # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐
"""
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4
# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""
# 文本转向量
#10+23+23
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)
#print(char_set,CHAR_SET_LEN)#63个字符
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        #print text
        idx = i * CHAR_SET_LEN + char2pos(c)
        #print i,CHAR_SET_LEN,char2pos(c),idx
        vector[idx] = 1
    return vector
#print(text2vec('0000'))

# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)
"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""
# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])#60*160=9600
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])#4*63=252

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y
####################################################################

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])#  x为特征
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])# y为label
keep_prob = tf.placeholder(tf.float32)  # dropout

# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    # 卷积中将1x784转换为28x28x1  [-1,,,]代表样本数量不变 [,,,1]代表通道数
    #reshape函数的作用是将tensor变换为参数shape的形式。
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    #假设 输入图片为 28*28
    # 第一个卷积层  [3, 3, 1, 32]代表 卷积核尺寸为3x3,1个通道,32个不同卷积核
    # 创建滤波器权值-->加偏置-->卷积-->池化
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    #28x28x1 与32个5x5x1滤波器 --> 28x28x32
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    #conv1 = tf.nn.relu(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1]+b_c1))

    # 28x28x32 -->14x14x32
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)



    # 第二层卷积层 卷积核依旧是5x5 通道为32   有64个不同的卷积核
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    #14x14x32 与64个5x5x32滤波器 --> 14x14x64
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
      #14x14x64 --> 7x7x64
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)


      # 第三层卷积层 卷积核依旧是5x5 通道为64   有64个不同的卷积核
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))

    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    # conv3的大小为7x7x64 转为1-D 然后做FC层
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 32 * 40, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    #8*32*40--> 1x10240
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    #FC层传播 10240 --> 1024
    #tf.matmul(​​X，W)表示x乘以W
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))

     # 使用Dropout层减轻过拟合,通过一个placeholder传入keep_prob比率控制
        # 在训练中,我们随机丢弃一部分节点的数据来减轻过拟合,预测时则保留全部数据追求最佳性能
    dense = tf.nn.dropout(dense, keep_prob)



    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # 将Dropout层的输出连接到一个Softmax层,得到最后的概率输出
    # out = tf.nn.softmax(out)
    return out
# 训练
def train_crack_captcha_cnn():
    import time
    start_time=time.time()
    output = crack_captcha_cnn()
    # loss
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    # 定义损失函数,依旧使用交叉熵  同时定义优化器  learning rate = 1e-4
#函数说明：sigmoid损失函数计算

# 参数1：labels

# 类型和logits一致

# 参数2：logits

# 类型 `float32` or `float64`.
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    # 定义评测准确率
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    #开始训练
    isTrain = True #来区分训练阶段和测试阶段，True 表示训练，False表示测试
    train_steps = 50 #表示训练的次数，例子中使用100
    checkpoint_steps = 5 #表示训练多少次保存一下checkpoints，例子中使用50
    checkpoint_dir = '.\\' #表示checkpoints文件的保存路径，例子中使用当前路径F:\\py3workspace\\train_captcha\\
    isAgainTrain=False #表示是否恢复保存的模型继续训练

    if isTrain:
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) #初始化所有变量
            step = 0
            f=open(checkpoint_dir+'acc.txt','w+')
            for step in range(train_steps):
                #step += 50
                batch_x, batch_y = get_next_batch(64)
                if isAgainTrain:
                    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
                _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
                print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),step, loss_)
                f.write(str(step+1)+', val_acc: '+str(loss_)+'\n')

                # 每100 step计算一次准确率
                if (step+1) % checkpoint_steps == 0 and step > 0:
                    batch_x_test, batch_y_test = get_next_batch(100)
                    acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                    print (u'***************************************************************第%s次的准确率为%s'%((step+1), acc))
                    #saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=step)
                    saver.save(sess, checkpoint_dir +"crack_capcha.model", global_step=step+1)
                    # 如果准确率大于50%,保存模型,完成训练
                    if acc > 0.5:                  ##我这里设了0.9，设得越大训练要花的时间越长，如果设得过于接近1，很难达到。如果使用cpu，花的时间很长，cpu占用很高电脑发烫。
                        saver.save(sess, checkpoint_dir +"crack_capcha.model", global_step=step+1)
                        print (time.time()-start_time)
                        break
    else:
        #output = crack_captcha_cnn()
        saver = tf.train.Saver(max_to_keep=1)
        sess = tf.Session()
        #latest_checkpoint自动获取最后一次保存的模型
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        #batch_x_test, batch_y_test = get_next_batch(100)
        #_, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
        #print(loss_)
        #quit()
        while(1):
            text, image = gen_captcha_text_and_image()
            image = convert2gray(image)
            image = image.flatten() / 255
            predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
            print(text_list)
            predict_text = text_list[0].tolist()
            print(predict_text)
            vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
            #print(vector)
            i = 0
            for t in predict_text:
                vector[i * 63 + t] = 1
                i += 1
                # break
            print(vector)
            print("正确: {}  预测: {}".format(text, vec2text(vector)))
            break
        sess.close()







            #output = crack_captcha_cnn()
            #saver = tf.train.Saver()
            #sess = tf.Session()
            #saver.restore(sess, tf.train.latest_checkpoint('F:\\py3workspace\\train_captcha\\'))

            # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            #tf.train.get_checkpoint_state可以用来检查是否有保存的checkpoint
            # if ckpt and ckpt.model_checkpoint_path:
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            # else:
            #     pass
            #print (u'***************************************************************第1次的准确率为%s'%(acc))

            #print(sess.run(w_out))
            #print(sess.run(b_out))
            #print(sess.run(loss_))
            #print(sess.run(acc))
            #saver.restore(sess, tf.train.latest_checkpoint('.'))

train_crack_captcha_cnn()
