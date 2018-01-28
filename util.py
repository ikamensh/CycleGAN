import math
import numpy as np

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1], 3),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img
    return image


def restore(img):
    img[img<-1] = -1
    img[img>1 ] = 1
    img += 1
    img *= 127.5
    return img.astype(np.uint8)


def preprocess(pics):
    temp = pics.astype(np.float32)
    temp /= 127.5
    temp -= 1
    return temp

import cv2


# temp = []
#
# fullname = "output/imgs" + '/' + "fakeA_89_8"
#
# temp.append(np.expand_dims(cv2.imread(fullname),axis=0))
#
# fullname = "output/imgs - Copy" + '/' + "inputA_36_8"
# temp.append(np.expand_dims(cv2.imread(fullname),axis=0))
#
# cv2.imwrite("temp", combine_images(np.vstack(temp)))