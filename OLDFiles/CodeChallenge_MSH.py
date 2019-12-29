# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 11:16:14 2018

@author: MuhammadSalman
"""



from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,median_filter
import sys
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import sem, t
from scipy import mean
from skimage import data, img_as_float
from skimage import exposure

#from pylab import *



def image_process(I,sigmaval):
    result1 = gaussian_filter(I, sigmaval)
    return result1





#path = 'C:/Users/55116867/Dropbox/codingchallenge/challengeFiles/'
path = 'C:/Users/MuhammadSalman/Dropbox/codingchallenge/challengeFiles/'
tiff_file = 'G-ex_P100_50deg_exp100_2017-05-18_15h25m02s936ms-2.tif'

I = plt.imread(path+tiff_file)
I = I/256
fig = plt.figure(figsize=(20,20))
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side

result = image_process(I, 2)
result = median_filter(I, size=6)

ax1.imshow(I)
ax2.imshow(I>20)
plt.show()



def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


# Load an example image
img = I.astype(np.uint8)

# Contrast stretching


# Equalization
img_eq = exposure.equalize_hist(img)

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.01)

# Display results
fig = plt.figure(figsize=(20,20))
axes = np.zeros((2, 4), dtype=np.object)
axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 4):
    axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
for i in range(0, 4):
    axes[1, i] = fig.add_subplot(2, 4, 5+i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
ax_img.set_title('Histogram equalization')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
ax_img.set_title('Adaptive equalization')

ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()





'''

if __name__ == '__main__':
   main()
'''
'''
def conf_int(I,confidence):
    
    bin_counts, bin_edges = np.histogram(I, bins='auto')
    b1 = np.argmax(bin_counts)
    sk_mean = np.mean([bin_edges[b1],bin_edges[b1+1]])
    #n = len(bin_edges)
    n = len(np.reshape(I,[np.size(I,0)*np.size(I,1),1]))
    #m = mean(data)
    std_err = sem(bin_edges)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    start = sk_mean - h
    end =sk_mean + h
    
    return start, end

start,end = conf_int(I,0.995)


##Histogram
import matplotlib.pyplot as plt
bin_counts, bin_edges, patches = plt.hist(I.ravel(), bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
b1 = np.argmax(bin_counts)
sk_mean = np.mean([bin_edges[b1],bin_edges[b1+1]])


'''
