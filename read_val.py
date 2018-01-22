import cv2
import numpy as np
import os
from util import preprocess

rainy = []

test_dir = 'input/rainy'

for filename in os.listdir(test_dir):

    fullname = test_dir + '/' + filename
    img = cv2.imread(fullname)
    rainy.append(np.expand_dims(img, axis=0))


rainy = np.vstack(rainy)

# test_val = rainy[:100]

rainy = preprocess(rainy)



sunny = []

sunny_dir = 'input/sunny'

for filename in os.listdir(sunny_dir):

    fullname = sunny_dir + '/' + filename
    img = cv2.imread(fullname)
    sunny.append(np.expand_dims(img, axis=0))


sunny = np.vstack(sunny)

# sunny_val = sunny[:100]

sunny = preprocess(sunny)

# test_l = np.zeros_like(test_pics)
# test_h = np.zeros_like(test_pics)
#
# for indx in range(test_pics.shape[0]):
#     hi, lo = split_hi_low(test_pics[indx])
#     test_h[indx] = hi
#     test_l[indx] = lo
#
# test_pics = None



