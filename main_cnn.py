# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
from model.CNN_plus import CNN
from utils.captcha_utils import get_data_x_y


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


# 计算模型在训练集的准确率
def acc(y_true,y_predict):

    acc = 0

    # 计算准确率
    for i in range(len(y_true)):
        if y_true[i] == y_predict[i]:
            acc += 1
    print(acc)
    return acc / len(y_true)


def my_interface():

    with tf.Graph().as_default():

        # 创建模型
        cnn = CNN()

        # 计算损失
        loss = cnn.losses()

        # 优化损失
        train_op = cnn.optimizer(loss)

        # 模型评估
        target_1, target_2, target_3, target_4 = cnn.get_text()

        # 模型的可视化
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir='./graph/cnn_plus',graph=tf.get_default_graph())


        # 模型训练

        with tf.Session() as session:

            # 参数初始化
            session.run(tf.global_variables_initializer())

            # 模型恢复
            cnn.restore(session=session,checkpoint_dir='./dlmodel/cnn_plus')

            # 生成数据
            # x,y,y_true = get_data_x_y(n=100,width=128,height=64)
            x, y, y_true = get_data_x_y(n=10000000, width=128, height=64)
            i = 1
            while True:

                # # 每次训练的时候生成数据，防止过拟合
                # x, y, y_true = get_data_x_y(n=100, width=128, height=64)


                # 喂给模型数据
                train_op_,loss_,target_1_, target_2_, target_3_, target_4_,summary_op_, = session.run([train_op,loss,target_1, target_2, target_3, target_4,summary_op],feed_dict={
                    cnn.input_x:x,
                    cnn.label:y
                })

                # 模型的可视化
                summary_writer.add_summary(summary_op_)


                y_predict = get_y_predict(target_1_, target_2_, target_3_, target_4_)
                print("第{}次训练的时候，模型的损失为{}，模型的准确率为{}".format(i,loss_,acc(y_true,y_predict)))

                # 若准确率大于0.9，则停止训练
                if acc(y_true,y_predict) > 0.9:
                    break
                    pass

                # 模型的保存
                if i % 2 == 0:

                    cnn.save(session=session,save_path='./dlmodel/cnn_plus/cnn.ckpt')

                    pass


                i += 1

                pass


            pass

        pass


    pass


my_interface()
































