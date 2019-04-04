#coding=utf-8
'''
有0-1,0-2,0-3 共3个分类器,每个分类器都是2分类器,其中0为0类,123为1类
投票规则：
对同一张图片,单个分类器判别为0且概率值>0.7,则投0一票,当0>=2时图片被判别为0类
否则负责判别为1类
'''

import os
import tensorflow as tf

from nets import inception_v3
from preprocessing import inception_preprocessing

slim = tf.contrib.slim
# 网络模型的输入图像有默认的尺寸,因此，我们需要先调整输入图片的尺寸
image_size = inception_v3.inception_v3.default_image_size

def single_predict (photo, model_dir, model_name, my_num_classes =2):
    '''
    :param photo: 输入图片路径
           model_dir: 模型存放绝对路径
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
            logits, _ = inception_v3.inception_v3(processed_images, num_classes=my_num_classes, is_training=False)

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

            coord.request_stop()
            coord.join(threads)

        return probabilities


def get_score(score, the_other_class):
    result_list =[]
    if score[0] > score[1]:
        result_list.append('0')
        result_list.append(float(score[0]))
    else:
        result_list.append(str(the_other_class))
        result_list.append(float(score[1]))
    return result_list   #[label, score]



def predict(photo, model_dir1, model_name1, model_dir2, model_name2, model_dir3, model_name3):
    score0_1 = single_predict(photo, model_dir1, model_name1)
    score0_2 = single_predict(photo, model_dir2, model_name2)
    score0_3 = single_predict(photo, model_dir3, model_name3)

    # 形式如[0_1label, 0_1score, 0_2label, 0_2score, 0_3label, 0_2score ]
    result_list = get_score(score0_1, 1)
    result_list.extend(get_score(score0_2, 2))
    result_list.extend(get_score(score0_3, 3))
    # print result_list

    # 判为0的结果>=2 则认为是 0 类
    num_0 = 0
    for i in range(len(result_list)):
        if result_list[i] == '0' and result_list[i + 1] > 0.70:
            num_0 += 1
    if num_0 >= 2:
        ensamble_result = '0'
    else:
        # 否则分类结果为1,2,3代表的负类
        ensamble_result = '1'

    return ensamble_result


#test
# photo = '/home/nicehija/PycharmProjects/images_test/my_test_image/or.jpg'
# model_dir1 = '/home/nicehija/PycharmProjects/flask_project/my_web/cnn_model/my_model_inceptionV3/3classifier_0andOthers/0_1'
# model_name1 = 'model.ckpt-6000'
# model_dir2 = '/home/nicehija/PycharmProjects/flask_project/my_web/cnn_model/my_model_inceptionV3/3classifier_0andOthers/0_2'
# model_name2 = 'model.ckpt-6000'
# model_dir3 = '/home/nicehija/PycharmProjects/flask_project/my_web/cnn_model/my_model_inceptionV3/3classifier_0andOthers/0_3'
# model_name3 ='model.ckpt-6000'
#
#
# r = predict(photo, model_dir1, model_name1, model_dir2, model_name2, model_dir3, model_name3)
# print  r