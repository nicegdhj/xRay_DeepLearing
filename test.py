import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from torchvision import models
import pandas as pd


# pred = np.random.randint(4, size=1000)
# label = np.random.randint(4, size=1000)
# marco_recall = recall_score(label, pred, average='macro')
# marco_accuracy = accuracy_score(label, pred)
# print(marco_recall)
# print(marco_accuracy)
a =range(1, 31)
print(a)

for i in range(5,31,5):
    print(i)