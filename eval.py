# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
from utils.captcha_utils import get_data_x_y_diy

# 数字，0~9，以字符串的方式存放到列表中
number = [str(i) for i in range(10)]

# 小写字母，a~z，利用ASCII码来转化为字符串存放到列表中
alphabet = [chr(97+i) for i in range(26)]

# 大写字母，A~Z，利用ASCII码来转化为字符串存放到列表中
ALPHABET = [chr(65+i) for i in range(26)]

# 所有验证码的charset
char_set = number + ALPHABET + alphabet

# 拿到y的预测值，即是4位验证码
def get_y_predict(target_1,target_2,target_3,target_4):

    # 验证码的4位
    text_1 = np.argmax(target_1,axis=1)
    text_2 = np.argmax(target_2, axis=1)
    text_3 = np.argmax(target_3, axis=1)
    text_4 = np.argmax(target_4, axis=1)

    text_all = []
    for i in range(text_1.shape[0]):

        # 最终预测的字符串
        text_item = char_set[text_1[i]] + char_set[text_2[i]] + char_set[text_3[i]] + char_set[text_4[i]]
        text_all.append(text_item)
        pass

    return text_all

def main(_):

    # 创建图
    with tf.Graph().as_default():

        graph = tf.get_default_graph()

        # 运行
        with tf.Session() as session:

            # 模型的恢复
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir='./dlmodel')
            if not(ckpt and ckpt.model_checkpoint_path):
                raise Exception("模型不存在")
            tf.logging.info("正在恢复模型!")

            saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
            saver.restore(sess=session,save_path=ckpt.model_checkpoint_path)


            # 你需要预测的数据
            data_x = ['00Oo','6563','f343']
            x,y,y_true = get_data_x_y_diy(data_x)

            # 拿到模型的输入Tensor
            input_x = graph.get_tensor_by_name('DNN/Input/input:0')

            # 拿到模型的输出Tensor
            predict = graph.get_tensor_by_name('DNN/Output/predict:0')

            output = session.run(predict,feed_dict={input_x:x})

            # 第一个字符
            target_1 = output[:, :62]
            # 第二个字符
            target_2 = output[:, 62:62 + 62]
            # 第三个字符
            target_3 = output[:, 62 + 62:62 + 62 + 62]
            # 第四个字符
            target_4 = output[:, 62 + 62 + 62:]

            print(get_y_predict(target_1,target_2,target_3,target_4))
            pass

        pass

    pass


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()



























