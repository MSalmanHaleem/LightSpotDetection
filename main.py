
##This code is designed for detecting smudges in the microscopic image
##Designed by Muhammad Salman Haleem



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
import numpy as np
from Filters import threshold_otsu,laplaceofGauss2D,white_tophat
from pwmorph import label





##Image Processing Function
def image_process(I,smudgesize):
    ##Top hat filtering for removing background noise
    result = white_tophat(I,'disk',12 )
    
    ##Detecting smudges for given size
    result22 = laplaceofGauss2D(result, (25,25),3.4)
    result22 = 1-result22
    val = threshold_otsu(result22)
    result1 = result22>val
    result2=result1#result2 = morphology.remove_small_objects(result1, 4)
    
    ##Labelling smudges detected based on watershed transform
    labels = label(result2)
    return labels



def main():

    path = sys.argv[1][0:len(sys.argv[1])]
    tiff_file = sys.argv[2][0:len(sys.argv[2])-4]
    opath = sys.argv[3][0:len(sys.argv[3])]

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
    with open(opath+tiff_file+'_rect.txt', 'w') as filehandle: 
        for regionnum in range(1,np.max(labels)+1):
    
            pp=np.argwhere(labels==regionnum)
        
            
            min_point=np.min(pp[:],axis=0)   
            max_point=np.max(pp[:],axis=0)   
            
            minr = min_point[0]
            minc = min_point[1]
            maxr = max_point[0]
            maxc = max_point[1]        
            
    
            
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=1)
            ax2.add_patch(rect)
            rect_list = [minr, minc, maxr, maxc]
            filehandle.write('%s\n' % rect_list)
    plt.show()

if __name__ == '__main__':
   main()



