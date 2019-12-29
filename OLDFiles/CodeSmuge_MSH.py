
##This code is designed for detecting smudges in the microscopic image
##Designed by Muhammad Salman Haleem



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from scipy.ndimage import white_tophat
import sys
import numpy as np
from skimage import  filters, morphology, feature,measure




##Image Processing Function
def image_process(I,smudgesize):
    ##Top hat filtering for removing background noise
    result = white_tophat(I,smudgesize*2)
    
    ##Detecting smudges for given size
    result22 = ndimage.gaussian_laplace(result, sigma=smudgesize/np.sqrt(2))
    result22 = 1-result22
    val = filters.threshold_otsu(result22)
    result1 = result22>val
    result2 = morphology.remove_small_objects(result1, 1)
    
    ##Labelling smudges detected based on watershed transform
    distance = ndimage.distance_transform_edt(result2)
    local_maxi = feature.peak_local_max(distance, indices=False, 
                                labels=result2)
    markers = ndimage.label(local_maxi)[0]
    labels = morphology.watershed(-distance, markers, mask=result2)
    return labels



def main():

    path = sys.argv[1][1:len(sys.argv[1])-1]
    tiff_file = sys.argv[2][1:len(sys.argv[2])-5]
    print(path)
    ##Tiff reading
    I = plt.imread(path+tiff_file+'.tif')
    I = I/256
    
    ##Image Processing and smudge detection
    smudgesize=6
    labels = image_process(I,smudgesize)
    
    ##Plotting 
    fig = plt.figure(figsize=(20,40))
    plt.gray()  # show the filtered result in grayscale
    ax1 = fig.add_subplot(121)  # left side 
    ax2 = fig.add_subplot(122)
    ax1.imshow(I)
    ax2.imshow(I)
    
    ##File writing for rectangular coordinates
    with open(path+tiff_file+'_rect.txt', 'w') as filehandle: 
        for region in measure.regionprops(labels):
            # take regions with large enough areas
            if region.area >= smudgesize:
        # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                ax2.add_patch(rect)
                rect_list = [minr, minc, maxr, maxc]
                filehandle.write('%s\n' % rect_list)
    
    plt.show()

if __name__ == '__main__':
   main()



