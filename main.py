
import tensorflow as tf
import numpy as np
from model.DNN import DNN
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

# 计算模型在测试集的准确率
def get_model_acc(n,session,dnn,train_op, losses, target_1, target_2, target_3, target_4):

    print("------------------------------------------------测试集--------------------------------------------")
    # 构造数据集
    x,y,y_true = get_data_x_y(n)

    # 喂给模型数据
    # 模型的训练
    train_op_, losses_, target_1_, target_2_, target_3_, target_4_ = session.run(
        [train_op, losses, target_1, target_2, target_3, target_4], feed_dict={
            dnn.inputs: x,
            dnn.label: y
    })

    # print(y_true)
    # y的预测值
    y_predict = get_y_predict(target_1_, target_2_, target_3_, target_4_)
    # print(y_predict)
    print("测试集，模型的损失为{}，模型的准去率为{}".format(losses_, acc(y_true, y_predict)))

    print("------------------------------------------------测试集--------------------------------------------")

    # 查看模型的准确率

    pass


def main(_):

    # 创建网络
    with tf.Graph().as_default():

        # 创建模型
        dnn = DNN()

        # 计算损失
        losses = dnn.losses()

        # 优化损失
        train_op = dnn.optimizer(losses)

        # 计算准确率
        target_1,target_2,target_3,target_4 = dnn.get_text()

        # 模型的可视化
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir='./graph',graph=tf.get_default_graph())

        # 执行、训练
        with tf.Session() as session:

            # 参数初始化
            session.run(tf.global_variables_initializer())

            # 喂给模型数据
            x, y, y_true = get_data_x_y(100)
            # x, y, y_true = get_data_x_y(10000000)
            # 模型的恢复
            dnn.restore(session=session,checkpoint_dir='./dlmodel')

            # 记录训练次数
            i = 0
            # 训练1000次
            while True:


                # 模型的训练
                train_op_,losses_,target_1_,target_2_,target_3_,target_4_,summary_op_ = session.run([train_op,losses,target_1,target_2,target_3,target_4,summary_op],feed_dict={
                    dnn.inputs:x,
                    dnn.label:y
                })

                # 添加模型可视化
                summary_writer.add_summary(summary_op_)


                # y的真实值
                # print(y_true)
                # y的预测值
                y_predict = get_y_predict(target_1_,target_2_,target_3_,target_4_)
                # print(y_predict)
                print("在第{}次训练的时候，模型的损失为{}，模型的准去率为{}".format(i,losses_,acc(y_true,y_predict)))

                # 模型的评估
                if i % 200 == 0:
                    get_model_acc(10,session,dnn,train_op, losses, target_1, target_2, target_3, target_4)
                    pass

                # 模型每训练100次即进行一次保存
                if i % 100 == 0:
                    dnn.save(session=session,save_path='./dlmodel/dnn.ckpt')
                    pass

                i += 1

                pass
            pass
        pass



    pass

if __name__ == '__main__':
    # 设置tf的日志级别
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()