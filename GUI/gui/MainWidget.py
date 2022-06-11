# -*- coding: utf-8 -*-


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from captcha.image import ImageCaptcha
import random
import tensorflow as tf
import numpy as np



class MainWidget(QWidget):

    # 构造方法
    def __init__(self):

        super(MainWidget, self).__init__()

        # 初始化数据
        self.__init_data__()

        # 初始化界面
        self.__init_view__()


        pass



    # 初始化数据
    def __init_data__(self):

        # 数字，0~9，以字符串的方式存放到列表中
        number = [str(i) for i in range(10)]

        # 小写字母，a~z，利用ASCII码来转化为字符串存放到列表中
        alphabet = [chr(97 + i) for i in range(26)]

        # 大写字母，A~Z，利用ASCII码来转化为字符串存放到列表中
        ALPHABET = [chr(65 + i) for i in range(26)]

        # 所有验证码的charset
        self.char_set = number + ALPHABET + alphabet


        pass

    # 初始化界面
    def __init_view__(self):

        # 界面的大小
        self.setFixedSize(640,480)

        # 设置标题
        self.setWindowTitle("验证码识别")

        # 声明垂直布局
        v_layout = QVBoxLayout(self)

        # todo 添加图片
        self.label_image = QLabel()
        # 声明图片
        jpg = QPixmap('./image/temp.png')
        # 添加
        self.label_image.setPixmap(jpg)
        self.label_image.setScaledContents(True)
        # 放置图片
        v_layout.addWidget(self.label_image)


        # 声明水平布局
        h_layout = QHBoxLayout(self)

        # todo 输入框
        self.textarea_captcha_content = QTextEdit(self)
        # 设置长和宽
        self.textarea_captcha_content.setMaximumHeight(60)
        self.textarea_captcha_content.setMaximumWidth(100)
        self.textarea_captcha_content.setText("在这里输入您想要生成的验证码(4位)...")
        h_layout.addWidget(self.textarea_captcha_content)

        # todo 生成验证码按钮
        self.btn_create_captcha = QPushButton('生成验证码')
        self.btn_create_captcha.setMaximumWidth(100)
        self.btn_create_captcha.setMaximumHeight(60)
        # 按钮的点击事件
        self.btn_create_captcha.clicked.connect(self.on_btn_create_captcha_click)
        h_layout.addWidget(self.btn_create_captcha)

        # todo 预测按钮
        self.btn_start = QPushButton("开始预测")
        self.btn_start.setMaximumHeight(60)
        self.btn_start.setMaximumWidth(100)
        h_layout.addWidget(self.btn_start)

        # todo 预测的结果
        self.textarea_answer = QTextEdit(self)
        self.textarea_answer.setText('')
        self.textarea_answer.setMaximumHeight(60)
        self.textarea_answer.setMaximumWidth(100)
        # 添加点击事件
        self.btn_start.clicked.connect(self.on_btn_start_click)
        h_layout.addWidget(self.textarea_answer)

        # 添加布局的嵌套
        v_layout.addLayout(h_layout)

        pass


    # 开始预测的点击事件
    def on_btn_start_click(self):

        # 创建图
        with tf.Graph().as_default():

            # 拿到模型的图
            graph = tf.get_default_graph()

            # 模型的恢复
            with tf.Session() as session:

                # 模型的恢复
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir='./model')
                # 判断模型是否存在model_checkpoint_path
                if not(ckpt and ckpt.model_checkpoint_path):
                    raise Exception("模型不存在。。。")
                    pass

                # 恢复
                saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
                saver.restore(sess=session,save_path=ckpt.model_checkpoint_path)
                # 喂给模型数据
                # 转化为灰度图片
                x = self.conver2gray(self.image_data)
                input_x = graph.get_tensor_by_name("CNN/input/input:0")
                # 预测

                # 获得模型预测的字符
                predict_1 = graph.get_tensor_by_name("CNN/predict_1/predict_1:0")
                predict_2 = graph.get_tensor_by_name("CNN/predict_2/predict_2:0")
                predict_3 = graph.get_tensor_by_name("CNN/predict_3/predict_3:0")
                predict_4 = graph.get_tensor_by_name("CNN/predict_4/predict_4:0")

                # 拿到结果
                predict_1_,predict_2_,predict_3_,predict_4_ = session.run([predict_1,predict_2,predict_3,predict_4],feed_dict={
                    input_x:x
                })

                answer = self.get_y_predict(predict_1_,predict_2_,predict_3_,predict_4_)

                # 把结果返回给Textarea
                self.textarea_answer.setText(answer)

                pass

            pass


        pass

    # 转化为灰度图片，然后拉平
    def conver2gray(self,image_data):
        # 转换为灰度图
        gray = 0.2989 * image_data[:,:,0] + 0.5870 * image_data[:,:,1] + 0.1140 * image_data[:,:,2]
        # 拉平
        return [gray.reshape(-1)]

    # 根据四个值，得到最后预测的结果
    def get_y_predict(self,predict_1_,predict_2_,predict_3_,predict_4_):

        text = ''
        # 拿到最有可能的字符
        text_1 = np.argmax(predict_1_,axis=1)
        text_2 = np.argmax(predict_2_,axis=1)
        text_3 = np.argmax(predict_3_,axis=1)
        text_4 = np.argmax(predict_4_,axis=1)

        text = self.char_set[text_1[0]] + self.char_set[text_2[0]] + self.char_set[text_3[0]] + self.char_set[text_4[0]]
        print(text)
        return text

    # 当点击生成验证码按钮的时候调用
    def on_btn_create_captcha_click(self):

        # 先拿到验证码输入框中的内容
        text = self.textarea_captcha_content.toPlainText()
        # 判断输入框中的内容
        if len(text) != 4:
            text = ''
            pass

        # 生成验证码
        self.get_captcha_image(text=text)

        png = QPixmap('./image/temp.png')

        self.label_image.setPixmap(png)

        # self.textarea_captcha_content.setText('')

        pass

    # 生成验证码
    def get_captcha_image(self,text=''):

        # 若用户没有指定生成特殊的验证码，则采用随机生成
        if text == '':
            for i in range(4):
                text += random.choice(self.char_set)
                pass
            pass
        # 生成的验证码文本，正确的验证码
        self.textarea_captcha_content.setText(text)

        # 生成图片
        image = ImageCaptcha(width=128,height=64)
        # 根据验证码的文本内容生成图片
        image = image.generate_image(text)

        self.image_data = np.array(image)

        # 保存图片
        image.save('./image/temp.png')

        pass

    pass




















