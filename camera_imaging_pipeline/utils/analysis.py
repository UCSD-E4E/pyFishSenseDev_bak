import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def display_data(img, p):
    print('The shape of the data is: ')
    print(img.shape)

    print('The max value of the data is: ')
    print(img.max())

    print('The min value of the data is: ')
    print(img.min())

    if p == True:
        print('The data is displayed below: ')
        print(img)


