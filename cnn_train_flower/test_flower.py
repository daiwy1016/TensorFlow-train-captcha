#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-11 16:11:34
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

path='F:/py3workspace/cnn_train_animal/flower_photos/'

cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
print(cate)

img=io.imread("F:/py3workspace/cnn_train_animal/cat.jpg")
print(type(img))

a=[1,2,3,4,5]
data=np.asarray(a,np.float32)
print(data.shape)
print(data.shape[0])

ratio=0.8
s=np.int(data.shape[0]*ratio)
print(s)
x_train=data[2:5]
print(x_train)

