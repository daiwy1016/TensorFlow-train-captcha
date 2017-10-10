#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-09 16:50:46
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
    	#方法返回一个列表，元组或字符串的随机项
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text
# 生成字符对应的验证码
def gen_captcha_text_and_image():
    while(1):
        image = ImageCaptcha()

        captcha_text = random_captcha_text()
        captcha_text = ''.join(captcha_text)

        captcha = image.generate(captcha_text)
        #image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

        captcha_image = Image.open(captcha)
        #captcha_image.show()
        captcha_image = np.array(captcha_image)
        if captcha_image.shape==(60,160,3):
            break

    return captcha_text, captcha_image
if __name__ == '__main__':
	# 测试
	text,image = gen_captcha_text_and_image()
	print(image)
	#返回的数组元素的平均值。平均取默认扁平阵列flatten(array)
	#上，否则在指定轴。 float64中间和返回值被用于整数输入。
	gray = np.mean(image,-1)
# >>> a = np.array([[1, 2], [3, 4]])
# >>> np.mean(a) # 将上面二维矩阵的每个元素相加除以元素个数（求平均数）
# 2.5
# >>> np.mean(a, axis=0) # axis=0，计算每一列的均值
# array([ 2.,  3.])
# >>> np.mean(a, axis=1) # 计算每一行的均值
# array([ 1.5,  3.5])
	print(gray)
	print(image.shape)
	print(gray.shape)
	f = plt.figure()
	ax = f.add_subplot(111)
	ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
	plt.imshow(image)
	plt.show()
