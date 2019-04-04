#coding=utf-8

import os
import tensorflow as tf

from nets import inception_v3
from preprocessing import inception_preprocessing

slim = tf.contrib.slim
# 网络模型的输入图像有默认的尺寸,因此，我们需要先调整输入图片的尺寸
image_size = inception_v3.inception_v3.default_image_size

def predict (photo, model_dir, model_name):
    '''
    :param photo: 输入图片路径
           model_dir: 模型存放绝度路径
           model_name: 模型名
    :return: 函数返回的一个list,包含了4类全部预测按概率大小排序，其中每一类是一个tuple

    加载训练好的的模型，输入图片后获取一个预测值
    '''


    photo_path = photo
    model_dir = model_dir

    with tf.Graph().as_default():
        filenames = [photo_path]
        filename_queue = tf.train.string_input_producer(filenames)

        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)

        image = tf.image.decode_jpeg(value, channels=3)
        # 对图片做缩放操作，保持长宽比例不变，裁剪得到图片中央的区域
        # 裁剪后的图片大小等于网络模型的默认尺寸
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)

        # 可以批量导入图像
        # 第一个维度指定每批图片的张数
        # 我们每次只导入一张图片
        processed_images = tf.expand_dims(processed_image, 0)

        # 创建模型，使用默认的arg scope参数
        # arg_scope是slim library的一个常用参数
        # 可以设置它指定网络层的参数，比如stride, padding 等等。
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, _ = inception_v3.inception_v3(processed_images, num_classes=4, is_training=False)

        # 我们在输出层使用softmax函数，使输出项是概率值
        probabilities = tf.nn.softmax(logits)

        # 创建一个函数，从checkpoint读入网络权值
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(model_dir, model_name),
            slim.get_variables_to_restore())

        # 初始化image_tf
        # init_op = tf.variables_initializer([image_tf])

        with tf.Session() as sess:
            # 加载权值
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            init_fn(sess)

            # 图片经过缩放和裁剪，最终以numpy矩阵的格式传入网络模型
            np_image, network_input, probabilities = sess.run([image,
                                                               processed_image,
                                                               probabilities])
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                                key=lambda x: x[1])]

            coord.request_stop()
            coord.join(threads)

        names = ['0', '1', '2', '3']
        result = []
        for i in range(4):
            # 输出4类的预测类别和相应的概率值
            index = sorted_inds[i]
            prob = ("%.4f" % probabilities[index])
            result.append((names[index], str(prob)))
        return result



#test

# photo = '/home/nicehija/PycharmProjects/beijingproject_tensorflow/slim/test_image/my/3/img20161122_10094063.jpg'
# model_dir = '/home/nicehija/PycharmProjects/flask_project/my_web/cnn_model/my_model_inceptionV3'
# model_name = 'model.ckpt-5000'
#
# a = predict(photo, model_dir, model_name)
# print (a)