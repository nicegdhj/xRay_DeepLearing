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

a = {10:1, 20:2}
for k,v in a.items():
    print(k)
    print(v)
