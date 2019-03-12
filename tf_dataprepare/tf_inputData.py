# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         18-12-24

Description:
不使用tfRecord的格式,直接放入内存读取
(因为使用后发现特别生成的tfrecord特别大...

参考:
https://zhuanlan.zhihu.com/p/30751039

关于buffer_size:
https://stackoverflow.com/questions/47781375/tensorflow-dataset-shuffle-large-dataset?rq=1
"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


def getFilesList(fileDir, classes):
    """
    :param fileDir: 目录
    :param classes:  数据类别
    :return:
    """
    image_list = []
    label_list = []
    for index_label, className in enumerate(classes):
        num = 0
        class_path = os.path.join(fileDir, className)
        for img in os.listdir(class_path):
            img_path = os.path.join(class_path, img)
            image_list.append(img_path)
            label_list.append(index_label)
            num = num + 1

        print('>>>>>>>>>>>> %s class %d images done >>>>>>>>>>>>' % (className, num))

    return image_list, label_list


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=0)  # tf.image.decode_jpeg可以改变channel
    ######################################
    # data argumentation should go to here
    ######################################
    image = tf.image.resize_images(image_decoded, [600, 600])
    # data_augmentation
    image = tf.random_crop(image, [448, 448, 1])  #
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # image = tf.image.resize_images(image_decoded, [500, 500])
    # image = tf.image.per_image_standardization(image)
    return image, label

#######TEST
train_dir = '/home/nicehjia/myProjects/xRay/xRayData/zipXrayImages/train'
val_dir = '/home/nicehjia/myProjects/xRay/xRayData/zipXrayImages/validation'
classes = ['0', '1', '2', '3']
BATCH_SIZE = 10

train_img, train_label = getFilesList(train_dir, classes)
val_img, val_label = getFilesList(val_dir, classes)
print('a')

# input data
## 定义source
# images = tf.constant(val_img)
# labels = tf.constant(val_label)
#
# dataset = tf.data.Dataset.from_tensor_slices((images, labels))
# dataset = dataset.map(_parse_function)
# # 关于buffer_size:https://stackoverflow.com/questions/47781375/tensorflow-dataset-shuffle-large-dataset?rq=1
# dataset = dataset.shuffle(buffer_size=10).repeat().batch(BATCH_SIZE)
#
# ## 消费source
# iterator = dataset.make_one_shot_iterator()
# image_batch, label_batch = iterator.get_next()

# with tf.Session() as sess:
#     i = 0
#     try:
#         while i < 1:
#             img_data, label_data = sess.run([image_batch, label_batch])
#             # just test one batch
#             for j in np.arange(BATCH_SIZE):
#                 print('label: %d' % label_data[j])
#                 print(np.shape(img_data))
#                 print(np.shape(img_data[j, :, :, :]))
#                 plt.imshow(img_data[j, :, :, :].squeeze(), cmap="gray")
#                 # plt.imshow(img_data[j, :, :, :])
#                 plt.show()
#
#             i += 1
#
#     except tf.errors.OutOfRangeError:
#         print('done!')
#
