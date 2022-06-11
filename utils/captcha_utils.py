# -*- coding: utf-8 -*-

# pip install captcha -i 清华云镜像
# captcha这个包主要是用来生成验证码图片的包
from captcha.image import ImageCaptcha
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt


# 数字，0~9，以字符串的方式存放到列表中
number = [str(i) for i in range(10)]

# 小写字母，a~z，利用ASCII码来转化为字符串存放到列表中
alphabet = [chr(97+i) for i in range(26)]

# 大写字母，A~Z，利用ASCII码来转化为字符串存放到列表中
ALPHABET = [chr(65+i) for i in range(26)]

# 所有验证码的charset
char_set = number + ALPHABET + alphabet


# 生成验证码的方法（图片加上标签），默认生成4位的验证码
def get_captcha_text_and_image(captcha_size = 4,text = '',width = 160,height=60):

    # 验证码的文本
    if text == '':
        for i in range(captcha_size):
            # 从char_set中随机选择字符进行组合
            text += random.choice(char_set)
            pass

    # 生成图片
    image = ImageCaptcha(width=width,height=height)

    # 根据验证码的文本生成验证码
    captcha = image.generate(text)

    # 把captcha转换成图片矩阵
    captcha_img = Image.open(captcha)
    captcha_img = np.array(captcha_img)

    # 查看图片
    # plt.imshow(captcha_img)
    # plt.show()

    return text,captcha_img

# 把图片做成灰度图片，像素值为160*60，减少计算量，因为识别验证码的时候因为不需要色彩信息
def convert2gray(img):

    if len(img.shape) > 2:
        # 将img转换为灰度图片
        gray = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]
        return gray
    else:
        return img

# 给定一个字符，返回one-hot位置：得到某个字符的索引
def get_char_index(c):
    # 拿到c的ASCII码
    k = ord(c)

    # 先判断是否是数字
    if k < 48 + 10:
        return k - 48

    # 如果是大写字母
    if k - 65 < 26:
        # 加10是因为前10个是数字
        return k - 65 + 10

    # 如果是小写字母
    if k - 97 < 26:
        # 加10是因为前10个是数字，加26是因为中间26个位置表示的是大写字母
        return k - 97 + 10 + 26

    pass

# 采用one-hot-encoding编码，来对每个验证码进行编码，比如验证码的4位字符中，每种字符由（10+26+26=62）中可能，那么就用62个01数字串来表示该字符，最后4个数字串进行拼接即可
def text_to_one_hot(text,captcha_size = 4):

    # 编码之后的值，先构造一个一维数组，共有4*62个0元素
    text_one_hot = np.zeros([captcha_size * len(char_set)])

    for i,item in enumerate(text):

        # 得到某个字符的索引，i*62相当于偏移值
        idx = i * 62 + get_char_index(item)
        # 把text_one_hot对应字母位置设置为1
        text_one_hot[idx] = 1

        pass

    # 最后返回text_one_hot编码值
    return text_one_hot

# 最后还需要one-hot解码，得到4位验证码
def one_hot_to_text(one_hot):

    # 验证码
    text = []
    for i,item in enumerate(one_hot):
        if item == 1:
            char_index = i % 62
            text.append(char_set[int(char_index)])
        pass

    return text

# 拿到传入模型的数据，默认为100条数据
def get_data_x_y_diy(data_list):

    x = []
    # one-hot encoding
    y = []
    # 真实的文本
    y_true = []

    for i in range(len(data_list)):
        # 拿到验证码的真实值文本和图片
        text,captcha_img = get_captcha_text_and_image(text=data_list[i])

        # 将图片转换为灰度值拉直后并放入x中
        x.append(convert2gray(captcha_img).reshape(-1))

        # y的实际文本
        y_true.append(text)

        # one-hot encoding之后的数据
        y.append(text_to_one_hot(text))

        pass

    return x,y,y_true

# 拿到传入模型的数据，默认为100条数据
def get_data_x_y(n = 100,width=160,height=60):

    x = []
    # one-hot encoding
    y = []
    # 真实的文本
    y_true = []

    for i in range(n):
        # 拿到验证码的真实值文本和图片
        text,captcha_img = get_captcha_text_and_image(width=width,height=height)

        # 将图片转换为灰度值拉直后并放入x中
        x.append(convert2gray(captcha_img).reshape(-1))

        # y的实际文本
        y_true.append(text)

        # one-hot encoding之后的数据
        y.append(text_to_one_hot(text))

        pass

    return x,y,y_true








