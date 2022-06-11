# -*- coding: utf-8 -*-


import tensorflow as tf

class CNN():


    # 构造方法
    def __init__(self):

        self.saver = None

        self.saver_parameters = {}

        self.interace()

        pass

    # 前向网络的构建
    def interace(self):

        # 建立网络，参数采用随机正态分布初始化
        with tf.variable_scope("CNN",initializer=tf.random_normal_initializer(mean=0,stddev=0.001)):

            # 输入
            with tf.variable_scope("input"):

                # 模型的输入
                self.input_x = tf.placeholder(shape=(None,64*128),dtype=tf.float32,name='input')

                # 模型的真实值
                self.label = tf.placeholder(shape=(None,62*4),dtype=tf.float32,name='label')

                pass

            # 重置输入，因为拉直后的数据无法做卷积
            with tf.variable_scope('input_change'):
                # [图片数量(-1表示未知),图片的长，宽，通道数]
                net = tf.reshape(self.input_x,shape=[-1,64,128,1])
                pass


            # 卷积
            with tf.variable_scope("conv_1"):

                # 卷积核
                filter_net = tf.get_variable(
                    name='conv_1',
                    # [h,w,c,卷积核数量]
                    shape=[2,2,1,32],
                    dtype=tf.float32
                )

                # 卷积
                net = tf.nn.conv2d(
                    # 输入
                    input=net,
                    # 卷积核
                    filter=filter_net,
                    # 滑动的步长[1,h,w,1]
                    strides=[1,1,1,1],
                    # 填充数据，VALID表示不填充，SAME表示填充
                    padding="SAME"
                )

                print("卷积1：[None,64,128,32]",net.get_shape())

                # todo 批归一化
                net = tf.layers.batch_normalization(net)

                pass

            # 激活
            with tf.variable_scope("relu"):
                net = tf.nn.relu(net)
                print("激活1：[None,64,128,32]",net.get_shape())

                pass

            # 卷积2
            with tf.variable_scope("conv_2"):

                # 卷积核
                filter_net = tf.get_variable(
                    name='conv_2',
                    # [h,w,输入的通道,卷积核数量]
                    shape=[4, 4, 32, 32],
                    dtype=tf.float32
                )

                # 卷积
                net = tf.nn.conv2d(
                    # 输入
                    input=net,
                    # 卷积核
                    filter=filter_net,
                    # 滑动的步长[1,h,w,1]
                    strides=[1, 1, 1, 1],
                    # 填充数据，VALID表示不填充，SAME表示填充
                    padding="SAME"
                )

                print("卷积2：[None,64,128,32]", net.get_shape())

                # todo 再次批归一化
                net = tf.layers.batch_normalization(net)

                pass

            # 激活2
            with tf.variable_scope("relu_2"):
                net = tf.nn.relu(net)
                print("激活2：[None,64,128,32]", net.get_shape())
                pass

            # 池化
            with tf.variable_scope("max_pool"):

                net = tf.nn.max_pool(
                    # 要池化的tensor对象
                    value=net,
                    # 池化的窗口，[batch=1,h,w,c=1]
                    ksize=[1,2,2,1],
                    # 滑动窗口移动，[batch=1,h,w,c=1]
                    strides=[1,2,2,1],
                    # 是否需要填充
                    padding="VALID"
                )

                print("池化1：[None,32,64,32]",net.get_shape())

                pass



            # 卷积3
            with tf.variable_scope("conv_3"):
                # 卷积核
                filter_net = tf.get_variable(
                    name="conv_3",
                    shape=[4,4,32,64],
                    dtype=tf.float32
                )

                # 卷积
                net = tf.nn.conv2d(
                    input=net,
                    filter = filter_net,
                    strides=[1,1,1,1],
                    padding="SAME"
                )

                print("卷积3：[None,32,64,64]",net.get_shape())

                # todo 批归一化
                net = tf.layers.batch_normalization(net)

                pass

            # 激活3
            with tf.variable_scope("relu_3"):
                net = tf.nn.relu(net)
                print("激活3：[None,32,64,64]", net.get_shape())
                pass


            # 卷积4
            with tf.variable_scope("conv_4"):
                # 卷积核
                filter_net = tf.get_variable(
                    name='conv_4',
                    shape=[4,4,64,64],
                    dtype=tf.float32
                )

                # 卷积
                net = tf.nn.conv2d(
                    input=net,
                    filter=filter_net,
                    strides=[1,1,1,1],
                    padding='SAME'
                )
                print("卷积4：[None,32,64,64]", net.get_shape())

                # todo 批归一化
                net = tf.layers.batch_normalization(net)

                pass

            # 激活4
            with tf.variable_scope("relu"):
                net = tf.nn.relu(net)
                print("激活4：[None,32,64,64]",net.get_shape())
                pass

            # 池化2
            with tf.variable_scope("max_pooling"):

                net = tf.nn.max_pool(
                    value=net,
                    ksize=[1,2,2,1],
                    strides=[1,2,2,1],
                    padding="VALID"
                )

                print("池化2：[None,16,32,64]",net.get_shape())

                pass

            # 卷积5
            with tf.variable_scope("conv_5"):

                filter_net = tf.get_variable(
                    name='conv_5',
                    shape=[4,4,64,128],
                    dtype=tf.float32
                )

                net = tf.nn.conv2d(
                    input=net,
                    filter=filter_net,
                    strides=[1,1,1,1],
                    padding='SAME'
                )

                print("卷积5：[None,16,32,128]",net.get_shape())

                # todo 批归一化
                net = tf.layers.batch_normalization(net)

                pass

            # 激活5
            with tf.variable_scope("relu"):
                net = tf.nn.relu(net)
                print("激活5：[None,16,32,128]", net.get_shape())
                pass

            # 卷积6
            with tf.variable_scope("conv_6"):
                filter_net = tf.get_variable(
                    name='conv_6',
                    shape=[4, 4, 128, 128],
                    dtype=tf.float32
                )

                net = tf.nn.conv2d(
                    input=net,
                    filter=filter_net,
                    strides=[1, 1, 1, 1],
                    padding='SAME'
                )

                print("卷积6：[None,16,32,128]", net.get_shape())

                # todo 批归一化
                net = tf.layers.batch_normalization(net)

                pass

            # 激活6
            with tf.variable_scope("relu"):
                net = tf.nn.relu(net)
                print("激活6：[None,16,32,128]", net.get_shape())
                pass

            # 池化3
            with tf.variable_scope("max_pooling"):
                net = tf.nn.max_pool(
                    value=net,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding="VALID"
                )

                print("池化3：[None,8,16,128]", net.get_shape())

                pass

            # 卷积7
            with tf.variable_scope("conv_7"):
                filter_net = tf.get_variable(
                    name='conv_7',
                    shape=[2, 2, 128, 256],
                    dtype=tf.float32
                )

                net = tf.nn.conv2d(
                    input=net,
                    filter=filter_net,
                    strides=[1, 1, 1, 1],
                    padding='SAME'
                )

                print("卷积7：[None,8,16,256]", net.get_shape())

                # todo 批归一化
                net = tf.layers.batch_normalization(net)

                pass

            # 激活7
            with tf.variable_scope("relu"):
                net = tf.nn.relu(net)
                print("激活7：[None,8,16,256]", net.get_shape())
                pass

            # 卷积8
            with tf.variable_scope("conv_8"):
                filter_net = tf.get_variable(
                    name='conv_8',
                    shape=[2, 2, 256, 256],
                    dtype=tf.float32
                )

                net = tf.nn.conv2d(
                    input=net,
                    filter=filter_net,
                    strides=[1, 1, 1, 1],
                    padding='SAME'
                )

                print("卷积8：[None,8,16,256]", net.get_shape())

                # todo 批归一化
                net = tf.layers.batch_normalization(net)

                pass

            # 激活8
            with tf.variable_scope("relu"):
                net = tf.nn.relu(net)
                print("激活8：[None,8,16,256]", net.get_shape())
                pass

            # 池化4
            with tf.variable_scope("max_pooling"):
                net = tf.nn.max_pool(
                    value=net,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding="VALID"
                )

                print("池化4：[None,4,8,256]", net.get_shape())

                pass

            # 卷积9
            with tf.variable_scope("conv_9"):
                filter_net = tf.get_variable(
                    name='conv_9',
                    shape=[2, 2, 256, 256],
                    dtype=tf.float32
                )

                net = tf.nn.conv2d(
                    input=net,
                    filter=filter_net,
                    strides=[1, 1, 1, 1],
                    padding='SAME'
                )

                print("卷积9：[None,4,8,256]", net.get_shape())

                # todo 批归一化
                net = tf.layers.batch_normalization(net)

                pass

            # 激活9
            with tf.variable_scope("relu"):
                net = tf.nn.relu(net)
                print("激活9：[None,4,8,256]", net.get_shape())
                pass

            # 卷积10
            with tf.variable_scope("conv_10"):
                filter_net = tf.get_variable(
                    name='conv_10',
                    shape=[2, 2, 256, 256],
                    dtype=tf.float32
                )

                net = tf.nn.conv2d(
                    input=net,
                    filter=filter_net,
                    strides=[1, 1, 1, 1],
                    padding='SAME'
                )

                print("卷积10：[None,4,8,256]", net.get_shape())

                # todo 批归一化
                net = tf.layers.batch_normalization(net)

                pass

            # 激活10
            with tf.variable_scope("relu"):
                net = tf.nn.relu(net)
                print("激活10：[None,4,8,256]", net.get_shape())
                pass

            # 池化5
            with tf.variable_scope("max_pooling"):
                net = tf.nn.max_pool(
                    value=net,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding="VALID"
                )

                print("池化5：[None,2,4,256]", net.get_shape())

                pass


            # 拉平
            with tf.variable_scope("flatten"):

                shape = net.get_shape()
                # 2*4*256
                dim_size = shape[1] * shape[2] * shape[3]

                # 拉平，-1表示样本数未知，可自动分配
                net = tf.reshape(net,shape=[-1,dim_size])

                # 全连接，输出为2048个神经元
                w_1 = tf.get_variable(
                    name='w1',
                    shape=[dim_size,2048],
                    dtype=tf.float32
                )

                b_1 = tf.get_variable(
                    name='b_1',
                    shape=[1,2048],
                    dtype=tf.float32
                )

                # 矩阵相乘
                net = tf.matmul(net,w_1) + b_1

                # todo 批归一化
                net = tf.contrib.layers.batch_norm(
                    inputs=net,
                    decay=0.9,
                    updates_collections=None,
                    is_training=True
                )

                print("全连接：[None,2048]",net.get_shape())

                pass

            # 预测
            # 验证码的第一个字符预测
            with tf.variable_scope("predict_1"):

                w_p_1 = tf.get_variable(
                    name='w_p_1',
                    shape=[2048,62],
                    dtype=tf.float32
                )

                b_p_1 = tf.get_variable(
                    name='b_p_1',
                    shape=[1,62],
                    dtype=tf.float32
                )

                predict_1 = tf.matmul(net,w_p_1) + b_p_1

                # todo 批归一化
                self.predict_1 = tf.contrib.layers.batch_norm(
                    inputs=predict_1,
                    decay=0.9,
                    updates_collections=None,
                    is_training=True
                )

                # 给predict_1一个名字
                self.predict_1 = tf.identity(self.predict_1,'predict_1')

                pass

            # 验证码的第二个字符预测
            with tf.variable_scope("predict_2"):

                w_p_2 = tf.get_variable(
                    name='w_p_2',
                    shape=[2048, 62],
                    dtype=tf.float32
                )

                b_p_2 = tf.get_variable(
                    name='b_p_2',
                    shape=[1, 62],
                    dtype=tf.float32
                )

                predict_2 = tf.matmul(net, w_p_2) + b_p_2

                # todo 批归一化
                self.predict_2 = tf.contrib.layers.batch_norm(
                    inputs=predict_2,
                    decay=0.9,
                    updates_collections=None,
                    is_training=True
                )

                self.predict_2 = tf.identity(self.predict_2, 'predict_2')

                pass

            # 验证码的第三个字符预测
            with tf.variable_scope("predict_3"):
                w_p_3 = tf.get_variable(
                    name='w_p_3',
                    shape=[2048, 62],
                    dtype=tf.float32
                )

                b_p_3 = tf.get_variable(
                    name='b_p_3',
                    shape=[1, 62],
                    dtype=tf.float32
                )

                predict_3 = tf.matmul(net, w_p_3) + b_p_3

                # todo 批归一化
                self.predict_3 = tf.contrib.layers.batch_norm(
                    inputs=predict_3,
                    decay=0.9,
                    updates_collections=None,
                    is_training=True
                )

                self.predict_3 = tf.identity(self.predict_3, 'predict_3')

                pass

            # 验证码的第四个字符预测
            with tf.variable_scope("predict_4"):
                w_p_4 = tf.get_variable(
                    name='w_p_4',
                    shape=[2048, 62],
                    dtype=tf.float32
                )

                b_p_4 = tf.get_variable(
                    name='b_p_4',
                    shape=[1, 62],
                    dtype=tf.float32
                )

                predict_4 = tf.matmul(net, w_p_4) + b_p_4

                # todo 批归一化
                self.predict_4 = tf.contrib.layers.batch_norm(
                    inputs=predict_4,
                    decay=0.9,
                    updates_collections=None,
                    is_training=True
                )

                self.predict_4 = tf.identity(self.predict_4, 'predict_4')

                pass


            pass


        pass

    # 计算模型的损失
    def losses(self):

        with tf.variable_scope('losses'):

            # y理论与y真实形成模型的损失
            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.label))

            loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predict_1,labels=self.label[:,:62]))

            loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predict_2,labels=self.label[:,62:62+62]))

            loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predict_3,labels=self.label[:,62+62:62+62+62]))

            loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predict_4,labels=self.label[:,62+62+62:]))

            loss = loss1 + loss2 + loss3 + loss4

            # 可视化
            tf.summary.scalar('loss',loss)
            pass

        return loss


    # 优化损失
    def optimizer(self,loss):

        with tf.variable_scope('train'):

            # 选择优化器，并设置学习率
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

            # 最小化损失
            train_op = optimizer.minimize(loss=loss)

            pass

        return train_op

    # 模型评估
    def get_text(self):
        # # 第一个字符
        # target_1 = self.output[:, :62]
        # # 第二个字符
        # target_2 = self.output[:, 62:62 + 62]
        # # 第三个字符
        # target_3 = self.output[:, 62 + 62:62 + 62 + 62]
        # # 第四个字符
        # target_4 = self.output[:, 62 + 62 + 62:]

        return self.predict_1, self.predict_2, self.predict_3, self.predict_4
        pass

    # 模型的保存
    def save(self,session,save_path):

        # 赋值
        if self.saver is None:
            self.saver = tf.train.Saver(**self.saver_parameters)
            pass

        # 模型的保存
        self.saver.save(sess=session,save_path=save_path)

        print("模型持久化完成！")

        pass

    # 模型的恢复
    def restore(self,session,checkpoint_dir):
        if self.saver is None:
            self.saver = tf.train.Saver(**self.saver_parameters)
            pass

        # 检查文件夹是否存在
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # 若存在，则进行模型恢复
            print("模型开始恢复。。。")
            self.saver.restore(sess=session,save_path=ckpt.model_checkpoint_path)
            pass
        else:
            # 否则打印提示
            print("没有找到训练好的模型，不能进行恢复")
        pass










    pass






































