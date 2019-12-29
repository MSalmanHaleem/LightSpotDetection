# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:50:39 2018

@author: 55116867
"""



import numpy as np
from scipy import signal
from StructElement import square,rectangle,diamond,disk





def threshold_otsu(image, nbins=256):

    if len(image.shape) > 2 and image.shape[-1] in (3, 4):
        msg = "threshold_otsu is expected to work correctly only for " \
              "grayscale images; image shape {0} looks like an RGB image"
        print(msg)

    # Check if the image is multi-colored or not
    if image.min() == image.max():
        raise ValueError("threshold_otsu is expected to work with images "
                         "having more than one color. The input image seems "
                         "to have just one color {0}.".format(image.min()))

    hist, bin_edges = np.histogram(image.ravel(), nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold



def laplaceofGauss2D(I,shape,sigma):
    """
    shape=(25,25) sigma =3.4
    """
    
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) ) * (1-((x*x + y*y) / (2.*sigma*sigma))) * (-1/(np.pi*(sigma**4)))
    I_out = signal.convolve2d(I, h, boundary='symm', mode='same')
    return I_out


def grey_erosion(I,struct_element,elesize):
    
    I_out = np.zeros([np.size(I,0),np.size(I,1)])
    padsize = int(elesize/2)
    I2 = np.pad(I,padsize,'symmetric') 
    for i in range(0,np.size(I,0)):
        for j in range(0,np.size(I,1)):
            ii,jj=np.where(struct_element==1)
            tmp = I2[i:i+elesize+1,j:j+elesize+1]
            tmp=  tmp[ii,jj]
            I_out[i,j] = np.min(tmp) 
    
    return I_out


def grey_dilation(I,struct_element,elesize):
    
    I_out = np.zeros([np.size(I,0),np.size(I,1)])
    padsize = int(elesize/2)
    I2 = np.pad(I,padsize,'symmetric') 
    for i in range(0,np.size(I,0)):
        for j in range(0,np.size(I,1)):
            ii,jj=np.where(struct_element==1)
            tmp = I2[i:i+elesize+1,j:j+elesize+1]
            tmp=  tmp[ii,jj]
            I_out[i,j] = np.max(tmp) 
    return I_out


def white_tophat(I,struct_element_name,elesize ):

    
    if struct_element_name=='square':
        struct_element = square(elesize)
    elif struct_element_name=='disk':
        struct_element = disk(int(elesize/2))
    tmp = grey_erosion(I,struct_element,elesize)
    tmp = grey_dilation(tmp, struct_element,elesize)
    tmp = I-tmp


    return tmp






