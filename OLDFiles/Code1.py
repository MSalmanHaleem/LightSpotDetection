# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 11:16:14 2018

@author: MuhammadSalman
"""



from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import sys
import numpy as np




def image_process(I,sigmaval):
    result1 = gaussian_filter(I, sigmaval)
    return result1

#path = 'C:/Users/55116867/Dropbox/codingchallenge/challengeFiles/'
#tiff_file = 'G-ex_P100_50deg_exp100_2017-05-18_15h25m02s936ms-1.tif'


def main():

    path = sys.argv[1][1:len(sys.argv[1])-1]
    tiff_file = sys.argv[2][1:len(sys.argv[2])-1]
    print(path)
    I = plt.imread(path+tiff_file)
    fig = plt.figure(figsize=(20,20))
    plt.gray()  # show the filtered result in grayscale
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    result = image_process(I, 2)
    ax1.imshow(I)
    ax2.imshow(result)
    plt.show()


if __name__ == '__main__':
   main()


'''

if __name__ == '__main__':
   main()
'''
