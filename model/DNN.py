# -*- coding: utf-8 -*-


import tensorflow as tf

class DNN():

    # 构造方法
    def __init__(self):


        # 模型的输入
        self.inputs = None

        # 模型的保存
        self.saver = None
        self.saver_parameters = {}

        # 第一隐藏层的神经元数目
        self.hidden_layer_1_nn_num = 1024

        # 第二隐藏层的神经元数目
        self.hidden_layer_2_nn_num = 1024

        # 第三隐藏层的神经元数目
        self.hidden_layer_3_nn_num = 2048

        # 第四隐藏层的神经元数目
        self.hidden_layer_4_nn_num = 2048

        # # 模型的输出
        # self.output = None

        # 构建前向网络
        self.interface()
        pass

    # 前向网络构建
    def interface(self):

        with tf.variable_scope("DNN"):

            # 输入层
            with tf.variable_scope('Input'):
                # 模型的输入
                # 每张图片的像素值为160*60=9600
                self.inputs = tf.placeholder(shape=(None,9600),dtype=tf.float32,name="input")

                # 模型的真实值
                self.label = tf.placeholder(shape=(None,248),dtype=tf.float32,name="label")

                pass

            # 第一隐藏层，1024个神经元
            with tf.variable_scope("Hidden_Layer_1"):

                # 参数
                w1 = tf.get_variable(name='w1',shape=(9600,self.hidden_layer_1_nn_num),dtype=tf.float32)

                b1 = tf.get_variable(name='b1',shape=(1,self.hidden_layer_1_nn_num),dtype=tf.float32)

                # 线性组合
                layer_1_output = tf.matmul(self.inputs,w1) + b1

                # 非线性激活
                layer_1_output_activate = tf.nn.sigmoid(layer_1_output)

                # todo 批归一化
                layer_1_output_activate = tf.contrib.layers.batch_norm(inputs=layer_1_output_activate,
                                                                       decay=0.9,
                                                                       updates_collections=None,
                                                                       is_training=True)
                pass


            # 第二隐藏层
            with tf.variable_scope("Hidden_Layer_2"):

                # 参数
                w2 = tf.get_variable(name='w2',shape=(self.hidden_layer_1_nn_num,self.hidden_layer_2_nn_num),dtype=tf.float32)

                b2 = tf.get_variable(name='b2',shape=(1,self.hidden_layer_2_nn_num),dtype=tf.float32)

                # 线性组合
                layer_2_output = tf.matmul(layer_1_output_activate,w2) + b2

                # 非线性激活
                layer_2_output_activate = tf.nn.sigmoid(layer_2_output)

                # todo 批归一化
                layer_2_output_activate = tf.contrib.layers.batch_norm(inputs=layer_2_output_activate,
                                                                       decay=0.9,
                                                                       updates_collections=None,
                                                                       is_training=True)

                pass

            # 第三隐藏层
            with tf.variable_scope("Hidden_Layer_3"):

                # 参数
                w3 = tf.get_variable(name='w3',shape=(self.hidden_layer_2_nn_num,self.hidden_layer_3_nn_num),dtype=tf.float32)

                b3 = tf.get_variable(name='b3',shape=(1,self.hidden_layer_3_nn_num),dtype=tf.float32)

                # 线性组合
                layer_3_output = tf.matmul(layer_2_output_activate,w3) + b3

                # 非线性激活
                layer_3_output_activate = tf.sigmoid(layer_3_output)

                # todo 批归一化
                layer_3_output_activate = tf.contrib.layers.batch_norm(inputs=layer_3_output_activate,
                                                                       decay=0.9,
                                                                       updates_collections=None,
                                                                       is_training=True)

                pass

            # 第四隐藏层
            with tf.variable_scope("Hidden_Layer_4"):
                # 参数
                w4 = tf.get_variable(name='w4', shape=(self.hidden_layer_3_nn_num, self.hidden_layer_4_nn_num),
                                     dtype=tf.float32)

                b4 = tf.get_variable(name='b4', shape=(1, self.hidden_layer_4_nn_num), dtype=tf.float32)

                # 线性组合
                layer_4_output = tf.matmul(layer_3_output_activate, w4) + b4

                # 非线性激活
                layer_4_output_activate = tf.sigmoid(layer_4_output)

                # todo 批归一化
                layer_4_output_activate = tf.contrib.layers.batch_norm(inputs=layer_4_output_activate,
                                                                       decay=0.9,
                                                                       updates_collections=None,
                                                                       is_training=True)
                pass


            # 输出层
            with tf.variable_scope("Output"):

                # 参数，248表示one-hot编码后得到的vector的长度
                w5 = tf.get_variable(name='w5',shape=(self.hidden_layer_4_nn_num,248),dtype=tf.float32)

                b5 = tf.get_variable(name='b5',shape=(1,248),dtype=tf.float32)

                # 线性组合
                self.output = tf.matmul(layer_4_output_activate,w5) + b5

                # 第一个字符
                target_1 = tf.nn.softmax(self.output[:,:62])
                # 第二个字符
                target_2 = tf.nn.softmax(self.output[:, 62:62+62])
                # 第三个字符
                target_3 = tf.nn.softmax(self.output[:, 62+62:62+62+62])
                # 第四个字符
                target_4 = tf.nn.softmax(self.output[:, 62+62+62:])

                # 拼接
                self.output = tf.concat([target_1,target_2,target_3,target_4],axis=1)


                # 给self.output一个名称
                self.output = tf.identity(self.output,'predict')


                pass
            pass

        pass

    # 计算模型的损失
    def losses(self):

        with tf.variable_scope("Losses"):

            # 根据y的理论值和y的真实值来去计算模型的损失 todo 多分类要选择softmax
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.label))

            # 可视化损失
            tf.summary.scalar('losses',loss)

            pass

        return loss

    # 优化损失
    def optimizer(self,loss):

        with tf.variable_scope("Train"):

            # 选择优化器
            optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)

            # 优化损失
            train_op = optimizer.minimize(loss=loss)

            pass
        return train_op

    # 模型评估
    def get_text(self):

        # 第一个字符
        target_1 = self.output[:, :62]
        # 第二个字符
        target_2 = self.output[:, 62:62 + 62]
        # 第三个字符
        target_3 = self.output[:, 62 + 62:62 + 62 + 62]
        # 第四个字符
        target_4 = self.output[:, 62 + 62 + 62:]

        return target_1,target_2,target_3,target_4
        pass

    # 模型的保存
    def save(self,session,save_path):

        # 模型的保存
        if self.saver is None:
            self.saver = tf.train.Saver(**self.saver_parameters)

        # 模型保存
        self.saver.save(sess=session,save_path=save_path)

        # 打印提示
        tf.logging.info("模型持久化完成")

        pass

    # 模型的恢复
    def restore(self,checkpoint_dir,session):

        # 拿到saver
        if self.saver is None:
            self.saver = tf.train.Saver(**self.saver_parameters)

        # 检查文件夹是否存在
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:

            # 打印提示
            tf.logging.info("开始恢复模型！！！")

            # 模型的恢复
            self.saver.restore(sess=session,save_path=ckpt.model_checkpoint_path)
            pass
        else:
            tf.logging.warn("没有找到训练好的模型，暂时不能进行模型的恢复！")
        pass



    pass





















