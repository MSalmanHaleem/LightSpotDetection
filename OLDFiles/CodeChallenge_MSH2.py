# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 11:16:14 2018

@author: MuhammadSalman
"""



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from scipy.ndimage import white_tophat
import sys
import numpy as np
from skimage import  filters, morphology, feature,measure


#from pylab import *



def image_process(I,smudgesize):
    #p1, p2 = np.percentile(img, (perc_min, perc_max))
    result1 = white_tophat(I,smudgesize)
    #result1 = exposure.rescale_intensity(img, in_range=(p1, p2))
    return result1





#path = 'C:/Users/55116867/Dropbox/codingchallenge/challengeFiles/'
path = 'C:/Users/MuhammadSalman/Dropbox/codingchallenge/challengeFiles/'
tiff_file = 'Wed_morningSession_nup555_647.1528885469405-2'

I = plt.imread(path+tiff_file+'.tif')
I = I/256
fig = plt.figure(figsize=(20,40))
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side 
ax2 = fig.add_subplot(122)
result = white_tophat(I,12)
result22 = ndimage.gaussian_laplace(result, sigma=3.4)
result22 = 1-result22
val = filters.threshold_otsu(result22)
result1 = result22>val
result2 = morphology.remove_small_objects(result1, 4)

distance = ndimage.distance_transform_edt(result2)

local_maxi = feature.peak_local_max(distance, indices=False, 
                            labels=result2)
markers = ndimage.label(local_maxi)[0]
labels = morphology.watershed(-distance, markers, mask=result2)
ax1.imshow(I)

ax2.imshow(I)
with open(path+tiff_file+'_rect.txt', 'w') as filehandle: 
    for region in measure.regionprops(labels):
        # take regions with large enough areas

    # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)
        ax2.add_patch(rect)
        rect_list = [minr, minc, maxr, maxc]
        filehandle.write('%s\n' % rect_list)
#ax4.imshow(labels,cmap=plt.cm.nipy_spectral, interpolation='nearest')
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
'''
#struct = ndimage.generate_binary_structure(2, 1)
#result = ndimage.morphological_gradient(I, structure=struct,size=(1,1))
result2 = morphology.erosion(result, morphology.disk(2))
'''
