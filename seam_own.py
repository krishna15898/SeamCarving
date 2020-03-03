import sys
import numpy as np
import pandas as pd
from imageio import imread, imwrite
import cv2
from matplotlib.pyplot import imshow

from scipy.ndimage import convolve

def getEnergy(im):
    dx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    dy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    en1 = convolve(im,dx)
    en2 = convolve(im,dy)
    energy = np.array([[np.sqrt(en1[i,j]**2+en2[i,j]**2)for j in range(im.shape[1])]for i in range(im.shape[0])])
    return energy

def get3denergy(im):
    energy = np.zeros((im.shape[0],im.shape[1]))
    for channel in range(3):
        dx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        dy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        en1 = convolve(im[:,:,channel],dx)
        en2 = convolve(im[:,:,channel],dy)
        energy  = energy + np.array([[np.sqrt(en1[i,j]**2+en2[i,j]**2)for j in range(im.shape[1])]for i in range(im.shape[0])])
    return energy

def getVerticalMask(energy):
    # initialising mask as combination of value and seam direction upwards 
    mask = np.array([[[energy[i,j],0] for j in range(energy.shape[1])] for i in range(energy.shape[0])])

    for i in range(1,energy.shape[0]):
        a = [mask[i-1,0,0],mask[i-1,1,0]]
        mask[i,0] = [mask[i,0,0]+min(a), a.index(min(a))]
        a = [mask[i-1,-2,0],mask[i-1,-1,0]]
        mask[i,-1] = [mask[i,-1,0]+min(a),a.index(min(a))-1]
        for j in range(1,energy.shape[1]-1):
            a = [mask[i-1,j-1,0],mask[i-1,j,0],mask[i-1,j+1,0]]
            mask[i,j] = [mask[i,j,0]+min(a),a.index(min(a))-1]
    return mask

# temp = is the cover for red
# kemp is the image in 3 layers
def reduceCols(img,mask,num):
    for col in range(num):
        img1 = np.zeros((img.shape[0],img.shape[1]-1,3));
        i = list(mask[-1,:,0]).index(min(mask[-1,:,0]));
        img1[-1] = np.delete(img[-1],i,0);
        for j in range(img.shape[0]-2,-1,-1):
            i = int(i+mask[j+1,i,1]);
            img1[j] = np.delete(img[j],i,0);
        del(mask);
        mask = getVerticalMask(get3denergy(img1));
        img = img1;
        del(img1);
    return img;

def getHorizontalMask(energy):
    # initialising mask as combination of energy values and seam direction leftwards 
    mask = np.array([[[energy[i,j],0] for j in range(energy.shape[1])] for i in range(energy.shape[0])])

    for j in range(1,energy.shape[1]):
        # first calculating for top and bottom pixels
        a = [mask[0,j-1,0],mask[1,j-1,0]]
        mask[0,j] = [mask[0,j,0]+min(a), a.index(min(a))]
        a = [mask[-2,j-1,0],mask[-1,j-1,0]]
        mask[-1,j] = [mask[-1,j,0]+min(a),a.index(min(a))-1]
        for i in range(1,energy.shape[0]-1):
            a = [mask[i-1,j-1,0],mask[i,j-1,0],mask[i+1,j-1,0]]
            mask[i,j] = [mask[i,j,0]+min(a),a.index(min(a))-1]
    return mask

def reduceRows(img,mask,num):
    temp = img[:,:,0];
    for count in range(num):
        img1 = np.zeros((img.shape[0]-1,img.shape[1],3));
        i = list(mask[:,-1,0]).index(min(mask[:,-1,0]));
        img1[:,-1] = np.delete(img[:,-1],i,0);
        for j in range(img.shape[1]-2,-1,-1):
            # iterating over columns to remove a pixel in each
            i = int(i+mask[i,j+1,1]);
            img1[:,j] = np.delete(img[:,j],i,0);
        del(mask);
        mask = getHorizontalMask(get3denergy(img1));
        img = img1;
        del(img1);
    return img;

# enter image name here
img_name = 'img12.tif'
image = imread(img_name);
energy = get3denergy(image);
verticalMask = getVerticalMask(energy);
horizontalMask = getHorizontalMask(energy);
# enter rows, colums to remove here

cols, rows = 2, 2

image1 = reduceCols(image,verticalMask,cols);
image1 = reduceRows(image1,horizontalMask,rows);
imwrite('seam_'+str(rows)+'_'+str(cols)+'_'+img_name,image1.astype('uint8'));