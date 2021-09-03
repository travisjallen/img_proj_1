#################################
# Travis Allen
# CS 6640
# Project 1
# 8.31.21
#################################

# import necessary libraries
from matplotlib.text import OffsetFrom
import numpy as np                  # do useful calculations
import matplotlib.pyplot as plt     # plot graphs/make figures
import skimage as sk                # lots
from skimage import io              # read/write images
from skimage import exposure        # histogram equalization

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

# read image(s)
cow = io.imread("cow.png")
grandview = io.imread("grandview.jpg")
ridge = io.imread("ridgeline.jpg")
legacy = io.imread("legacy_bridge.jpg")

# transform image with user defined function
# grey_cow = color2grey(cow)
grey_grandview = color2grey(grandview)
# grey_ridge = color2grey(ridge)
# grey_legacy = color2grey(legacy)

# save images as .png
# io.imsave("grey_cow.png",grey_cow)
# io.imsave("grey_grandview.png",grey_grandview)
# io.imsave("grey_ridge.png",grey_ridge)
# io.imsave("grey_legacy.png",grey_legacy)

# Display the images to the user - note that values are float64 not uint8
# fig, (ax0,ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=4)
# ax0.imshow(cow)
# ax1.imshow(grandview)
# ax2.imshow(ridge)
# ax3.imshow(legacy)
# ax0.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
# ax1.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
# ax2.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
# ax3.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
# plt.show()
# plt.savefig('original_images.png')

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
def grey2hist(img,num_bins,plot,title,filename):
    # inputs are:
    # img: greyscale image to make histogram of
    # num_bins: number of bins in histogram
    # plot: should the function make a histogram? if yes, set plot=1
    # title: title on plot
    # filename: name to save image as. specifiy file type

    # determine the shape of the image
    image_size = img.shape
    num_rows = image_size[0]
    num_cols = image_size[1]
    
    # determine max intensity in image (for robustness)
    max_intensity = np.amax(img)
    
    # determine bin widths from max intensity and number of bins
    hist = np.zeros((num_bins,2))
    width = max_intensity/num_bins
    
    # loop through hist to fill first column with bin delimiters
    for u in range(num_bins):
        hist[u,0] = (u)*(width)

    # find number of counts in each bin
        # print(i)
    for j in range(num_rows):
        for k in range(num_cols):
            for i in range(num_bins-1):
                # check intensity at (j,k), compare to bin limits
                if (img[j,k] > hist[i,0]) and (img[j,k] < hist[i+1,0]):
                    hist[i,1]+=1
    
    # stack all columns of img end-to-end to create 1D array
    # note that order doesn't matter here because we are only
    # interested in counts
    vals = img.reshape(num_cols*num_rows,1)

    # now print and show a histogram of the image if asked for
    if plot == 1:
        plt.bar(hist[:,0],hist[:,1],width=width,align='edge')
        plt.title(title)
        plt.savefig(filename, dpi=150)
        plt.show()
        
    
    # return the bins and counts of the histogram from our calculation
    return hist

# read greyscale image
# gimg = io.imread("grey_cow.png", plugin='matplotlib')

# call histogram function, specify plot = 1 to show histogram
# hist_cow = grey2hist(gimg,25,1,'Histogram of Grey Cow','grey_cow_hist.png')
# hist_grand = grey2hist(grey_grandview,25,1,'Histogram of Cottonwood Gulch','grandview_hist.png')
# hist_legacy = grey2hist(grey_legacy,25,1,'Histogram of Legacy Bridge','legacy_hist.png')
# hist_ridge = grey2hist(grey_ridge,25,1,'Histogram of Ridgeline','ridge_hist.png')

# show histogram bin lower delimiters and counts
# print(hist)

#######################################################################
# 3. Regions and components: Define a function that performs 
# double-sided (high and low) thresholding on images to define
# regions, visualize results (and histograms) on several images.
# Perform flood fill and connected component on these 
# thresholded images.  Remove connected components that are smaller 
# than a certain size (you specify).  Visualize the results as a 
# color image (different colors for different regions).  

def threshold(grey_img,delimiter):
    # grey_img: greyscale image to threshold
    # delimiter: intensity value at which to split image on scale from 1 to 0
    
    # determine the shape of the image
    image_size = grey_img.shape
    num_rows = image_size[0]
    num_cols = image_size[1]

    # determine max intensity
    max_intensity = np.amax(grey_img)

    # scale max intensity by delimiter
    cutoff = delimiter * max_intensity

    # threshold each pixel
    for j in range(num_rows):
        for k in range(num_cols):
            # check intensity at (j,k), compare to bin limits
            if (grey_img[j,k] < cutoff):
                grey_img[j,k] = 0
            else:
                grey_img[j,k] = 1
    return grey_img

# thresh_cow = threshold(grey_cow,0.5)
thresh_grandview = threshold(grey_grandview,0.5)
# thresh_ridge = threshold(grey_ridge,0.5)
# thresh_legacy = threshold(grey_legacy,0.5)

# io.imsave("thresh_cow.png",thresh_cow)
# io.imsave("thresh_grandview.png",thresh_grandview)
# io.imsave("thresh_ridge.png",thresh_ridge)
# io.imsave("thresh_legacy.png",thresh_legacy)

# now perform connected component analysis

# import necessary tools
from skimage import measure

# define components as parts of the image that have intensity
# greater than 0.5
components = grey_grandview>0.5

# give labels to each conected component
labels = measure.label(components)

# What does this do?
components_labels = measure.label(components,background=0)

# show results side by side
plt.figure(figsize=(9, 3.5))

# first show just the connected components
plt.subplot(131)
plt.imshow(components, cmap='gray')
plt.axis('off')

# then show the labels
plt.subplot(132)
plt.imshow(labels, cmap='nipy_spectral')
plt.axis('off')

# then show the connected components labels
plt.subplot(133)
plt.imshow(components_labels, cmap='nipy_spectral')
plt.axis('off')

plt.tight_layout()
plt.show()

#######################################################################
# 4. Histogram equalization:  Perform histogram equalization on a 
# selection of images, show the histograms before and after 
# equalization, and comment on the visual results.   Perform 
# adaptive and/or local histogram equalization on a variety of 
# images.  Identify all important parameters, experiment with 
# (vary) those parameters, and report results on selection of 
# images of different types (photos, medical, etc.).

# # read the original image
# img = io.imread("cow.png")

# # transform image with user defined function
# grey_img = color2grey(img)

# # save the new image
# io.imsave("grey_cow.png",grey_img)

# # read the new image
# gimg = io.imread("grey_cow.png")

# # build histogram with new grey image
# g_hist = grey2hist(gimg,15,1,'grey_cow_hist.png')

# # now we have all of the pieces necessary to perform histogram equalization

# # perform histogram equalization using skimage exposure.equalize_hist()
# HE_gimg = exposure.equalize_hist(gimg)

# # save the histogram equalized image
# io.imsave("HE_grey_cow.png",HE_gimg)

# # make histogram of histogram equalized image
# HE_g_hist = grey2hist(HE_gimg,15,1,'HE_grey_cow_hist.png')

