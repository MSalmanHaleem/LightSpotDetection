# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 18:06:29 2018

@author: MuhammadSalman
"""

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

from skimage import  filters, morphology, feature,measure





def image_process(I,tophatval):
    result = white_tophat(I,tophatval)
    val = filters.threshold_otsu(result)
    result = result>val
    result2 = morphology.remove_small_objects(result, 1)
    
    distance = ndimage.distance_transform_edt(result2)
    
    local_maxi = feature.peak_local_max(distance, indices=False, 
                                labels=result2)
    markers = ndimage.label(local_maxi)[0]
    labels = morphology.watershed(-distance, markers, mask=result2)
    return labels

#path = 'C:/Users/55116867/Dropbox/codingchallenge/challengeFiles/'
#tiff_file = 'G-ex_P100_50deg_exp100_2017-05-18_15h25m02s936ms-1.tif'


def main():

    path = sys.argv[1][1:len(sys.argv[1])-1]
    tiff_file = sys.argv[2][1:len(sys.argv[2])-1]
    print(path)
    I = plt.imread(path+tiff_file)
    I = I/256
    
    labels = image_process(I,12)
    fig = plt.figure(figsize=(20,40))
    plt.gray()  # show the filtered result in grayscale
    ax1 = fig.add_subplot(121)  # left side 
    ax2 = fig.add_subplot(122)
    ax1.imshow(I)
    ax2.imshow(I)
    
    for region in measure.regionprops(labels):

        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax2.add_patch(rect)
    
    plt.show()

if __name__ == '__main__':
   main()


'''

if __name__ == '__main__':
   main()
'''
