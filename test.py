import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import recall_score

# from_dir = '/home/nicehjia/myProjects/data/chestXray14/chesXray14_4classes'
#
# imageLabelMap = {}
# for _, _, imageList in os.walk(from_dir):
#     # dir_list = ['images01', 'images02', 'images03', 'images04', 'images05', 'images06', 'images07', 'imagses09',
#     #             'images10', 'images11']
#     count = 0
#     dir_list = ['images01']
#     for dir in dir_list:
#         dir = os.path.join(from_dir, dir)
#         for img in os.listdir(dir):
#             count += 1
#             if count % 5000 == 0:
# #                 print('>>>>>>>> %s  images done>>>>>>>' % count)
# from torchvision import get_image_backend
# print(get_image_backend())

a = [0.9098181227066026, 0.9109790581228723, 0.9106754288601556, 0.9153727521598317, 0.9139915366902184, 0.9133009289554118, 0.9140374046402213, 0.9145077323216845, 0.9045329545172166, 0.8995966105345118, 0.9043772830435611, 0.9023054598391412, 0.9016148521043347, 0.898852421165108, 0.8960899902258815, 0.8905651283474285, 0.8730837447298451, 0.8511644557555477, 0.8195298069197446]
print(list(map(lambda x: float('%.4f' % x), a)))