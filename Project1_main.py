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
grey_cow = color2grey(cow)
grey_grandview = color2grey(grandview)
grey_ridge = color2grey(ridge)
grey_legacy = color2grey(legacy)

# save images as .png
# io.imsave("grey_cow.png",grey_cow)
# io.imsave("grey_grandview.png",grey_grandview)
# io.imsave("grey_ridge.png",grey_ridge)
# io.imsave("grey_legacy.png",grey_legacy)

# Display the images to the user - note that values are float64 not uint8
# plt.figure(figsize=(12, 4))
# plt.subplot(141)
# plt.imshow(cow)
# plt.axis('off')

# plt.subplot(142)
# plt.imshow(grandview)
# plt.axis('off')

# plt.subplot(143)
# plt.imshow(ridge)
# plt.axis('off')

# plt.subplot(144)
# plt.imshow(legacy)
# plt.axis('off')

# plt.tight_layout()
# plt.savefig('original_images.png')
# plt.show()

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
    hist = np.zeros((num_bins+1,2))
    width = max_intensity/num_bins
    
    # loop through hist to fill first column with bin delimiters
    for u in range(num_bins+1):
        hist[u,0] = (u)*(width)

    # find number of counts in each bin
        # print(i)
    for j in range(num_rows):
        for k in range(num_cols):
            for i in range(num_bins):
                # check intensity at (j,k), compare to bin limits
                if (img[j,k] >= hist[i,0]) and (img[j,k] <= hist[i+1,0]):
                    hist[i,1]+=1
    
    # stack all columns of img end-to-end to create 1D array
    # note that order doesn't matter here because we are only
    # interested in counts
    vals = img.reshape(num_cols*num_rows,1)

    # now print and show a histogram of the image if asked for
    if plot == 1:
        # make figure
        plt.figure(figsize=(6, 4))
        
        # plot image next to histogram
        plt.subplot(121)
        plt.imshow(img,cmap='gray')
        plt.title('Greyscale Image')
        plt.axis('off')

        plt.subplot(122)
        plt.bar(hist[:,0],hist[:,1],width=width,align='edge')
        plt.title(title)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        
    
    # return the bins and counts of the histogram from our calculation
    return hist, width

# read greyscale image
# gimg = io.imread("grey_cow.png", plugin='matplotlib')

# call histogram function, specify plot = 1 to show histogram
# hist_cow,width = grey2hist(grey_cow,25,1,'Histogram of Grey Cow','grey_cow_hist.png')
# hist_grand,width = grey2hist(grey_grandview,25,1,'Histogram of Cottonwood Gulch','grandview_hist.png')
# hist_legacy,width = grey2hist(grey_legacy,25,1,'Histogram of Legacy Bridge','legacy_hist.png')
# hist_ridge,width = grey2hist(grey_ridge,25,1,'Histogram of Ridgeline','ridge_hist.png')

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
    new_img = np.zeros([num_rows,num_cols])

    # determine max intensity
    max_intensity = np.amax(grey_img)

    # scale max intensity by delimiter
    cutoff = delimiter * max_intensity

    # threshold each pixel
    for j in range(num_rows):
        for k in range(num_cols):
            # check intensity at (j,k), compare to bin limits
            if (grey_img[j,k] < cutoff):
                new_img[j,k] = 0
            else:
                new_img[j,k] = 1
    return new_img

# threshold at 3 levels for cow, plot results, make histograms
thresh_cow_25 = threshold(grey_cow,0.25)
thresh_cow_50 = threshold(grey_cow,0.5)
thresh_cow_75 = threshold(grey_cow,0.75)
# thresh_cow_25_h,width = grey2hist(thresh_cow_25,10,0,'na','na')
# thresh_cow_50_h,width = grey2hist(thresh_cow_50,10,0,'na','na')
# thresh_cow_75_h,width = grey2hist(thresh_cow_75,10,0,'na','na')

# print(thresh_cow_25_h)

# plt.figure(figsize=(8, 5))
        
# plt.subplot(231)
# plt.imshow(thresh_cow_25,cmap='gray')
# plt.title('Cutoff at 0.25(max intensity)')
# plt.axis('off')

# plt.subplot(232)
# plt.imshow(thresh_cow_50,cmap='gray')
# plt.title('Cutoff at 0.5(max intensity)')
# plt.axis('off')

# plt.subplot(233)
# plt.imshow(thresh_cow_75,cmap='gray')
# plt.title('Cutoff at 0.75(max intensity)')
# plt.axis('off')

# plt.subplot(234)
# plt.bar(thresh_cow_25_h[:,0],thresh_cow_25_h[:,1],width=width,align='edge')

# plt.subplot(235)
# plt.bar(thresh_cow_50_h[:,0],thresh_cow_50_h[:,1],width=width,align='edge')

# plt.subplot(236)
# plt.bar(thresh_cow_75_h[:,0],thresh_cow_75_h[:,1],width=width,align='edge')

# plt.tight_layout()
# plt.savefig('cow_thresh_diag', dpi=300)
# plt.show()
        
# threshold at 3 levels for grandview, plot results ###############################################
thresh_grandview_25 = threshold(grey_grandview,0.25)
thresh_grandview_50 = threshold(grey_grandview,0.5)
thresh_grandview_75 = threshold(grey_grandview,0.75)
# thresh_grand_25_h,width = grey2hist(thresh_grandview_25,10,0,'na','na')
# thresh_grand_50_h,width = grey2hist(thresh_grandview_50,10,0,'na','na')
# thresh_grand_75_h,width = grey2hist(thresh_grandview_75,10,0,'na','na')


# plt.figure(figsize=(8, 5))
        
# plt.subplot(231)
# plt.imshow(thresh_grandview_25,cmap='gray')
# plt.title('Cutoff at 0.25(max intensity)')
# plt.axis('off')

# plt.subplot(232)
# plt.imshow(thresh_grandview_50,cmap='gray')
# plt.title('Cutoff at 0.5(max intensity)')
# plt.axis('off')

# plt.subplot(233)
# plt.imshow(thresh_grandview_75,cmap='gray')
# plt.title('Cutoff at 0.75(max intensity)')
# plt.axis('off')

# plt.subplot(234)
# plt.bar(thresh_grand_25_h[:,0],thresh_grand_25_h[:,1],width=width,align='edge')

# plt.subplot(235)
# plt.bar(thresh_grand_50_h[:,0],thresh_grand_50_h[:,1],width=width,align='edge')

# plt.subplot(236)
# plt.bar(thresh_grand_75_h[:,0],thresh_grand_75_h[:,1],width=width,align='edge')


# plt.tight_layout()
# plt.savefig('grandview_thresh_diag', dpi=300)
# plt.show()

# threshold at 3 levels for ridge, plot results
thresh_ridge_25 = threshold(grey_ridge,0.25)
thresh_ridge_50 = threshold(grey_ridge,0.5)
thresh_ridge_75 = threshold(grey_ridge,0.75)
# print("here")
# thresh_ridge_25_h,width = grey2hist(thresh_ridge_25,10,0,'na','na')
# print("here")
# thresh_ridge_50_h,width = grey2hist(thresh_ridge_50,10,0,'na','na')
# print("here")
# thresh_ridge_75_h,width = grey2hist(thresh_ridge_75,10,0,'na','na')
# print("here")

# plt.figure(figsize=(8, 5))
        
# plt.subplot(231)
# plt.imshow(thresh_ridge_25,cmap='gray')
# plt.title('Cutoff at 0.25(max intensity)')
# plt.axis('off')

# plt.subplot(232)
# plt.imshow(thresh_ridge_50,cmap='gray')
# plt.title('Cutoff at 0.5(max intensity)')
# plt.axis('off')

# plt.subplot(233)
# plt.imshow(thresh_ridge_75,cmap='gray')
# plt.title('Cutoff at 0.75(max intensity)')
# plt.axis('off')

# plt.subplot(234)
# plt.bar(thresh_ridge_25_h[:,0],thresh_ridge_25_h[:,1],width=width,align='edge')

# plt.subplot(235)
# plt.bar(thresh_ridge_50_h[:,0],thresh_ridge_50_h[:,1],width=width,align='edge')

# plt.subplot(236)
# plt.bar(thresh_ridge_75_h[:,0],thresh_ridge_75_h[:,1],width=width,align='edge')

# plt.tight_layout()
# plt.savefig('ridge_thresh_diag', dpi=300)
# plt.show()

# threshold at 3 levels for legacy, plot results
thresh_legacy_25 = threshold(grey_legacy,0.25)
thresh_legacy_50 = threshold(grey_legacy,0.5)
thresh_legacy_75 = threshold(grey_legacy,0.75)
# print("here")
# thresh_legacy_25_h,width = grey2hist(thresh_legacy_25,10,0,'na','na')
# print("here")
# thresh_legacy_50_h,width = grey2hist(thresh_legacy_50,10,0,'na','na')
# print("here")
# thresh_legacy_75_h,width = grey2hist(thresh_legacy_75,10,0,'na','na')
# print("here")

# plt.figure(figsize=(8, 5))
          
# plt.subplot(231)
# plt.imshow(thresh_legacy_25,cmap='gray')
# plt.title('Cutoff at 0.25(max intensity)')
# plt.axis('off')

# plt.subplot(232)
# plt.imshow(thresh_legacy_50,cmap='gray')
# plt.title('Cutoff at 0.5(max intensity)')
# plt.axis('off')

# plt.subplot(233)
# plt.imshow(thresh_legacy_75,cmap='gray')
# plt.title('Cutoff at 0.75(max intensity)')
# plt.axis('off')

# plt.subplot(234)
# plt.bar(thresh_legacy_25_h[:,0],thresh_legacy_25_h[:,1],width=width,align='edge')

# plt.subplot(235)
# plt.bar(thresh_legacy_50_h[:,0],thresh_legacy_50_h[:,1],width=width,align='edge')

# plt.subplot(236)
# plt.bar(thresh_legacy_75_h[:,0],thresh_legacy_75_h[:,1],width=width,align='edge')

# plt.tight_layout()
# plt.savefig('legacy_thresh_diag', dpi=300)
# plt.show()

# io.imsave("thresh_cow.png",thresh_cow)
# io.imsave("thresh_grandview.png",thresh_grandview)
# io.imsave("thresh_ridge.png",thresh_ridge)
# io.imsave("thresh_legacy.png",thresh_legacy)

# now perform connected component analysis ############################################

# import necessary tools
from skimage import measure
from skimage import morphology

# find each conected component in each thresholded image, show side by side.
# remove connected components smaller than 64 connected pixels with 
# morphology.area_closing()

# cow #######################################################################
cc_cow_25 = measure.label(thresh_cow_25)
cc_cow_50 = measure.label(thresh_cow_50)
cc_cow_75 = measure.label(thresh_cow_75)

ac_cow_25 = morphology.area_closing(thresh_cow_25,area_threshold=512)
ac_cow_50 = morphology.area_closing(thresh_cow_50,area_threshold=512)
ac_cow_75 = morphology.area_closing(thresh_cow_75,area_threshold=512)

plt.figure(figsize=(8, 5))

plt.subplot(231)
plt.imshow(cc_cow_25, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(232)
plt.imshow(cc_cow_50, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(233)
plt.imshow(cc_cow_75, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(234)
plt.imshow(ac_cow_25, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.imshow(ac_cow_50, cmap='gray')
plt.axis('off')

plt.subplot(236)
plt.imshow(ac_cow_75, cmap='gray')
plt.axis('off')

plt.savefig('cow_cc.png', dpi=300)

plt.tight_layout()
plt.show()

# grandview #######################################################################
cc_grand_25 = measure.label(thresh_grandview_25)
cc_grand_50 = measure.label(thresh_grandview_50)
cc_grand_75 = measure.label(thresh_grandview_75)

ac_grand_25 = morphology.area_closing(thresh_grandview_25,area_threshold=512)
ac_grand_50 = morphology.area_closing(thresh_grandview_50,area_threshold=512)
ac_grand_75 = morphology.area_closing(thresh_grandview_75,area_threshold=512)

plt.figure(figsize=(8, 5))

plt.subplot(231)
plt.imshow(cc_grand_25, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(232)
plt.imshow(cc_grand_50, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(233)
plt.imshow(cc_grand_75, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(234)
plt.imshow(ac_grand_25, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.imshow(ac_grand_50, cmap='gray')
plt.axis('off')

plt.subplot(236)
plt.imshow(ac_grand_75, cmap='gray')
plt.axis('off')

plt.savefig('grand_cc.png', dpi=300)

plt.tight_layout()
plt.show()

# ridge #######################################################################
cc_ridge_25 = measure.label(thresh_ridge_25)
cc_ridge_50 = measure.label(thresh_ridge_50)
cc_ridge_75 = measure.label(thresh_ridge_75)

ac_ridge_25 = morphology.area_closing(thresh_ridge_25,area_threshold=512)
ac_ridge_50 = morphology.area_closing(thresh_ridge_50,area_threshold=512)
ac_ridge_75 = morphology.area_closing(thresh_ridge_75,area_threshold=512)

plt.figure(figsize=(8, 5))

plt.subplot(231)
plt.imshow(cc_ridge_25, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(232)
plt.imshow(cc_ridge_50, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(233)
plt.imshow(cc_ridge_75, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(234)
plt.imshow(ac_ridge_25, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.imshow(ac_ridge_50, cmap='gray')
plt.axis('off')

plt.subplot(236)
plt.imshow(ac_ridge_75, cmap='gray')
plt.axis('off')

plt.savefig('ridge_cc.png', dpi=300)

plt.tight_layout()
plt.show()

# legacy #######################################################################
cc_legacy_25 = measure.label(thresh_legacy_25)
cc_legacy_50 = measure.label(thresh_legacy_50)
cc_legacy_75 = measure.label(thresh_legacy_75)

ac_legacy_25 = morphology.area_closing(thresh_legacy_25,area_threshold=512)
ac_legacy_50 = morphology.area_closing(thresh_legacy_50,area_threshold=512)
ac_legacy_75 = morphology.area_closing(thresh_legacy_75,area_threshold=512)

plt.figure(figsize=(8, 5))

plt.subplot(231)
plt.imshow(cc_legacy_25, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(232)
plt.imshow(cc_legacy_50, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(233)
plt.imshow(cc_legacy_75, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(234)
plt.imshow(ac_legacy_25, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.imshow(ac_legacy_50, cmap='gray')
plt.axis('off')

plt.subplot(236)
plt.imshow(ac_legacy_75, cmap='gray')
plt.axis('off')

plt.savefig('legacy_cc.png', dpi=300)

plt.tight_layout()
plt.show()

# now show thresholded images above removed connected-component images:

# cow ####################################################################
plt.figure(figsize=(8, 5))

plt.subplot(231)
plt.imshow(thresh_cow_25, cmap='gray')
plt.axis('off')

plt.subplot(232)
plt.imshow(thresh_cow_50, cmap='gray')
plt.axis('off')

plt.subplot(233)
plt.imshow(thresh_cow_75, cmap='gray')
plt.axis('off')

plt.subplot(234)
plt.imshow(ac_cow_25, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.imshow(ac_cow_50, cmap='gray')
plt.axis('off')

plt.subplot(236)
plt.imshow(ac_cow_75, cmap='gray')
plt.axis('off')

plt.savefig('cow_cc_compare.png', dpi=300)

plt.tight_layout()
plt.show()

# grandview ##############################################################
plt.figure(figsize=(8, 5))

plt.subplot(231)
plt.imshow(thresh_grandview_25, cmap='gray')
plt.axis('off')

plt.subplot(232)
plt.imshow(thresh_grandview_50, cmap='gray')
plt.axis('off')

plt.subplot(233)
plt.imshow(thresh_grandview_75, cmap='gray')
plt.axis('off')

plt.subplot(234)
plt.imshow(ac_grand_25, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.imshow(ac_grand_50, cmap='gray')
plt.axis('off')

plt.subplot(236)
plt.imshow(ac_grand_75, cmap='gray')
plt.axis('off')

plt.savefig('grand_cc_compare.png', dpi=300)

plt.tight_layout()
plt.show()

# ridge ##################################################################
plt.figure(figsize=(8, 5))

plt.subplot(231)
plt.imshow(thresh_ridge_25, cmap='gray')
plt.axis('off')

plt.subplot(232)
plt.imshow(thresh_ridge_50, cmap='gray')
plt.axis('off')

plt.subplot(233)
plt.imshow(thresh_ridge_75, cmap='gray')
plt.axis('off')

plt.subplot(234)
plt.imshow(ac_ridge_25, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.imshow(ac_ridge_50, cmap='gray')
plt.axis('off')

plt.subplot(236)
plt.imshow(ac_ridge_75, cmap='gray')
plt.axis('off')

plt.savefig('ridge_cc_compare.png', dpi=300)

plt.tight_layout()
plt.show()

# legacy #################################################################
plt.figure(figsize=(8, 5))

plt.subplot(231)
plt.imshow(thresh_legacy_25, cmap='gray')
plt.axis('off')

plt.subplot(232)
plt.imshow(thresh_legacy_50, cmap='gray')
plt.axis('off')

plt.subplot(233)
plt.imshow(thresh_legacy_75, cmap='gray')
plt.axis('off')

plt.subplot(234)
plt.imshow(ac_legacy_25, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.imshow(ac_legacy_50, cmap='gray')
plt.axis('off')

plt.subplot(236)
plt.imshow(ac_legacy_75, cmap='gray')
plt.axis('off')

plt.savefig('legacy_cc_compare.png', dpi=300)

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

