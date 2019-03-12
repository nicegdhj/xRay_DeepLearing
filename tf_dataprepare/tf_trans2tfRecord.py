# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         18-12-23

Description:

将图片装换为tfrecord格式

#参考:
http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
http://blog.csdn.net/chaipp0607/article/details/72960028

#步骤:
encode阶段
img-->example(Features to describe data in example)-->tfrecord

decode阶段
reader-->decoder-->images（使用了queue）

接下来步骤
images-->preprocess-->batch input

-----------------------------------------------------------
未使用

"""

import os
import numpy as np
import tensorflow as tf
import skimage.io as io
import cv2


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert2tfRecord(imageDir, save_dir, tf_file_name):
    """
    转化为tfrecord
    最常用的形式 image_name: label
    用于train set 和 validation set
    """
    classes = ['0', '1', '2', '3']

    filename = os.path.join(save_dir, tf_file_name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    print('\nTransform start......')
    num = 0

    for index, className in enumerate(classes):
        class_path = os.path.join(imageDir, className)

        for img in os.listdir(class_path):
            try:

                if num % 100 == 0:
                    print('>>>>>>>>>>>>>>>%d 张图片已经转换完成...>>>>>>>>>>>>' % num)
                img_path = os.path.join(class_path, img)
                image = io.imread(img_path)
                image_raw = image.tostring()  # type(image) must be array!
                label = int(index)


                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': int64_feature(label),
                    'image_raw': bytes_feature(image_raw)}))
                writer.write(example.SerializeToString())
                num = num + 1
                print(num, index)
            except IOError as e:
                print('Could not read:', img)
                print('error: %s' % e)
                print('Skip it!\n')

    writer.close()
    print('Transform done..  共转换 %d 张图片' % num)


def convert2tfRecord_v2(images, ids, save_dir, tf_file_name):
    """
    转化为tfrecord
    与常见的形式相比，主要还是label变成了文件名字，其余的不变
    用于test set
    -------
    images: 每一张图片的路径
    ids: 每张图片对应的名字
    """
    filename = os.path.join(save_dir, tf_file_name + '.tfrecords')
    n_samples = len(ids)

    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], n_samples))

    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in range(0, n_samples):
        try:
            if (i + 1) % 1000 == 0:
                print
                '已转换' + str(i + 1) + '/' + str(n_samples)
            image = io.imread(images[i])  # type(image) must be array!
            image_raw = image.tostring()
            id = str(ids[i])

            example = tf.train.Example(features=tf.train.Features(feature={
                'ids': bytes_feature(id),
                'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())

        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' % e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')


def data_augmentation(image, reshape_size):
    '''
    每张图片都必须经历的步骤: 归一化
    之后data augmentation
    ————————————
    :param reshape_size: 是一个list，[height, width, channel]
    '''

    image = tf.random_crop(image, reshape_size)  # randomly crop the image size to 24 x 24 x 3
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

    return image


def label_to_one_hot(label_batch, n_classes):
    """
    将label变为one hot形式
    计算loss时：
    tf.nn.sparse_softmax_cross_entropy_with_logits，不需要对label做one-hot变化
    tf.nn.softmax_cross_entropy_with_logits(), 需要one-hot
    """

    label_batch = tf.one_hot(label_batch, depth=n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])
    return label_batch


def check_decode_images(images, labels, batch_size, images_dir):
    '''
    检查tfrecord文件是否能够被正确decode，以及对应上标签
    ————————————
    输入是一个batch
    '''
    label_dict = {1: 'airplane', 2: 'automobile', 3: 'bird', 4: 'cat', 5: 'deer', 6: 'dog', 7: 'frog', 8: 'horse',
                  9: 'ship', 10: 'truck'}
    for i in range(batch_size):
        image = images[i, :, :, :]
        label = labels[i]
        label_name = label_dict[int(label)]
        cv2.imwrite(images_dir + '/' + label_name + '_' + str(i) + '.png', image)


def read_and_decode(tfrecords_file, batch_size, is_shuffle=None, label_one_hot=None):
    '''
    读取tfrecord 并且解码， batch
    image: 4D tensor - [batch_size, width, height, channel]
    label: 1D tensor - [batch_size]
    '''
    # 第一步，指定读入的queue
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    # 第二步，指定一个reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # 第三步decode过程
    # tf.parse_single_example解析器，可以将Example协议内存块(protocol buffer)解析为张量
    feature = {'label': tf.FixedLenFeature([], tf.int64),
               'image_raw': tf.FixedLenFeature([], tf.string)}
    img_features = tf.parse_single_example(serialized_example, features=feature)

    image = tf.decode_raw(img_features['image_raw'], tf.uint8)  # RGB图像应该是8bit，像素值0~255
    image = tf.cast(image, tf.float32)  # 转化到tf.float32
    # image = data_augmentation # 可以选择data augmentation
    image = tf.reshape(image, [32, 32, 3])  # reshape
    image = tf.image.per_image_standardization(image)  # 归一化

    label = tf.cast(img_features['label'], tf.int32)
    if is_shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=64,
            capacity=20000,
            min_after_dequeue=3000)
    else:
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=2000)

    if label_one_hot:
        label_batch = label_to_one_hot(label_batch)
    else:
        label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch


# Test
if __name__ == '__main__':
    # encode test
    trainDir = '/home/nicehjia/myProjects/xRaydata/zipXrayImages/train'
    valDir = '/home/nicehjia/myProjects/xRaydata/zipXrayImages/validation'
    saveDir = '/home/nicehjia/myProjects/xRaydata/zipXrayImages/'
    convert2tfRecord(trainDir, saveDir, tf_file_name='xRayTrain')
    convert2tfRecord(valDir, saveDir, tf_file_name='xRayValidation')

    # # decode test
    # BATCH_SIZE = 10
    #
    #
    # tf_file_path = '/Users/nicehjia/my_projects/remote_GPU_projects/limu_learnDL/data/tf_cifar10_train/cifar10_train.tfrecords'
    # check_data_dir = '/Users/nicehjia/my_projects/remote_GPU_projects/limu_learnDL/data/check_data'
    #
    # image_batch, label_batch = read_and_decode(tf_file_path, BATCH_SIZE)
    # with tf.Session() as sess:
    #     i = 0
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #     try:
    #         while not coord.should_stop() and i < 1:
    #             # just plot one batch size
    #             image, label = sess.run([image_batch, label_batch])
    #             check_decode_images(image, label, BATCH_SIZE, check_data_dir)
    #             i += 1
    #
    #     except tf.errors.OutOfRangeError:
    #         print('done!')
    #     finally:
    #         coord.request_stop()
    #     coord.join(threads)

    # encode testset
    # images_path = '/Users/nicehjia/my_projects/deepLearning_projects/learnDL_Limu/data/my_data_cifar10_train/airplane'
    # tf_save_dir = '/Users/nicehjia/my_projects/deepLearning_projects/learnDL_Limu/data/check_data'
    # image_list, id_list = get_file_testset(images_path)
    # convert_to_tfrecord_testset(image_list, id_list, tf_save_dir, 'test_testset')
