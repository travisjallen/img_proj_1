#################################
# Travis Allen
# CS 6640
# Project 1
# 8.31.21
#################################

# import necessary libraries
from ctypes import BigEndianStructure
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from skimage import io

#######################################################################
# 1: Preliminaries:
# You will need to be able to read images from a 
# file (e.g. jpg or png), convert, as needed, to 
# greyscale (make a function for this, using numpy 
# "dot" command), display images (with a greyscale 
# colormap), save images (for use in your report). 

# define function to convert to greyscale with dot product
def color2grey(img):
    grey = [0.2126, 0.7152, 0.0722] # HD TV standard for RGB luminance
    return np.dot(img[...,:3],grey)

# read image
img = io.imread("cow.png")

# transform image with user defined function
grey_img = color2grey(img)

# Display the image to the user - note that values are float64 not uint8
# plt.figure
# plt.imshow(grey_img) # , cmap="gray")
# plt.title("Grey Cow")
# plt.show()

# Save the image as .png
# io.imsave("grey_cow.png",grey_img)

#######################################################################
# 2: Build a histogram:  Write a function (from scratch, 
# using iterators in numpy) that takes a greyscale 
# image/array and returns a 2D array where the first 
# column entries are the histogram bin values (start 
# values) and the second column are the bin counts.
# Display the resulting histogram using a bar chart 
# from matplotlib.   Display histograms for a couple of 
# different images and describe how they relate to what 
# you see in the image (e.g. what regions/objects are 
# what part of the histogram).  Thresholding (below) 
# can help with this. 

# define function to make a histogram from a greyscale image
def grey2hist(img,num_bins,plot):
    # determine the size of the image
    image_size = img.shape
    num_rows = image_size[0]
    num_cols = image_size[1]
    
    # determine max intensity in image (for robustness)
    max_intensity = np.amax(img)
    
    # determine bin widths from max intensity and number of bins
    hist = np.zeros((num_bins,2))
    
    # loop through hist to fill first column with bin delimiters
    for u in range(num_bins):
        hist[u,0] = (u)*(max_intensity/num_bins)

    # find number of counts in each bin
        # print(i)
    for j in range(num_rows):
        for k in range(num_cols):
            for i in range(num_bins-1):
                # check intensity at (i,j), compare to bin limits
                if (img[j,k] > hist[i,0]) and (img[j,k] < hist[i+1,0]):
                    hist[i,1]+=1
    
    # stack all columns of img end-to-end to create 1D array
    # note that order doesn't matter here because we are only
    # interested in counts
    vals = img.reshape(num_cols*num_rows,1)

    # now print and show a histogram of the image
    n, bins, patches = plt.hist(vals, num_bins, density=True, facecolor='g', alpha=0.75)
    if plot == 1:
        plt.title('Histogram')
        plt.show()
    
    # return the bins and counts of the histogram from our calculation
    return hist

# read greyscale image
gimg = io.imread("grey_cow.png", plugin='matplotlib')

# call histogram function, specify plot = 1 to show histogram
# hist = grey2hist(gimg,10,1)

# show histogram bin lower delimiters and counts
# print(hist)

#######################################################################
# 4. Histogram equalization:  Perform histogram equalization on a 
# selection of images, show the histograms before and after 
# equalization, and comment on the visual results.   Perform 
# adaptive and/or local histogram equalization on a variety of 
# images.  Identify all important parameters, experiment with 
# (vary) those parameters, and report results on selection of 
# images of different types (photos, medical, etc.).

# define histogram equaliztion function
def hist_eq(img,num_bins):
    # make histogram of image with function from above
    hist = grey2hist(img,num_bins,0)

    # find shape of image
    im_size = img.shape
    num_rows = im_size[0]
    num_cols = im_size[1]

    # find number of elements in image
    MN = num_rows * num_cols

    # create array to store probabilities
    prob = np.zeros((num_bins,1))

    # find probability of occurence of intensity levels
    for i in range(num_bins):
        prob[i] = hist[i,1]/MN

    # create array to store transformed intensities, called s
    s = np.array((num_bins,1))

    # calculate transformed intensities, store in s
    for j in range(num_bins):
        for k in range(j):
            print(j)
            s[j] += num_bins*prob[k]

    # replace existing intensities with appropriate new intensities
    for j in range(num_rows):
        for k in range(num_cols):
            for i in range(num_bins-1):
                # check intensity at (i,j), compare to bin limits
                if (img[j,k] > hist[i,0]) and (img[j,k] < hist[i+1,0]):
                    img[j,k] = s[i]

    plt.figure 
    plt.imshow(img)
    plt.show()
    return

hist_eq(gimg,10)