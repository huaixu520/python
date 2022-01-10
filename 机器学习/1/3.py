import cv2
import os
import numpy as np


def read_dir(directory_name):
    array_of_img = []
    for filename in os.listdir("./" + directory_name):
        img = cv2.imread(directory_name + "/" + filename).ravel()
        array_of_img.append(img)
        img_array = np.array(array_of_img)
    print(img_array)


def cos_sim(a, b):
    a_norm = np.linalg.norm(a, ord=2, keepdims=True)
    b_norm = np.linalg.norm(b, ord=2, keepdims=True)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos


if __name__ == "__main__":
    read_dir("img")
    array_cos_new = []
    for i in range(11):
        for j in range(11):
            img_file1 = './img/%d.jpg' % i
            img_file2 = './img/%d.jpg' % j
            img1 = cv2.imread(img_file1).ravel()
            img2 = cv2.imread(img_file2).ravel()
            cos = cos_sim(img1, img2).ravel()
            array_cos_new.append(cos)
            a = np.array(array_cos_new)
            a = np.round(a, 10)
    array = a.reshape(11, 11)
    print(array)
    print(array.shape)
