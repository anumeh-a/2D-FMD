
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.utils import np_utils
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import os
from tensorflow.keras.optimizers import SGD

import keras
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.layers.core import Reshape

import random
import h5py
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from tensorflow.keras.optimizers import SGD
from keras.models import model_from_json
from keras.models import Model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
import tensorflow as tf
from numpy import genfromtxt
import pickle
import pandas as pd 
import math

my_data = genfromtxt('imcdf1.csv', delimiter=',')
split_images_data = np.hsplit(my_data, 20)
my_mask = genfromtxt('masksdrive1.csv', delimiter=',')
split_images_mask = np.hsplit(my_mask, 20)
my_datatest = genfromtxt('imcdf12.csv', delimiter=',')
split_images_dataset = np.hsplit(my_datatest, 20)
my_masktest = genfromtxt('drive2.csv', delimiter=',')
split_images_masktest = np.hsplit(my_masktest, 20)


shapeimg = split_images_data[0].shape
Nimgs = 20
height = shapeimg[0]
width = shapeimg[1]
imgss = np.empty((Nimgs,height,width))
groundTruths = np.empty((Nimgs,height,width))
imgsst = np.empty((Nimgs,height,width))
groundTruthst = np.empty((Nimgs,height,width))

for i in range(20):
    imgss[i] = Image.fromarray(split_images_data[i])
    groundTruths[i] = Image.fromarray(split_images_mask[i])
    imgsst[i] = Image.fromarray(split_images_dataset[i])
    groundTruthst[i] = Image.fromarray(split_images_masktest[i])

def get_datasets(imgs,groundTruth,train_test="null"):
    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    imgs = np.reshape(imgs,(Nimgs,1,height,width))
    assert(imgs.shape == (Nimgs,1,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    return imgs, groundTruth

imgs_train, groundTruth_train = get_datasets(imgss,groundTruths,"train")
imgs_test, groundTruth_test = get_datasets(imgsst,groundTruthst,"test")
print("saving test datasets")

#prepare the mask in the right shape for the Unet
def masks_Unet(masks):
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,2))
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if  masks[i,j] == 0:
                new_masks[i,j,0]=1
                new_masks[i,j,1]=0
            else:
                new_masks[i,j,0]=0
                new_masks[i,j,1]=1
    return new_masks


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
    return pred_images

def get_data_training(train_imgs_original,
                      train_groundTruth,
                      patch_height,
                      patch_width,
                      N_subimgs,):

    train_masks = train_groundTruth #masks always the same
    train_imgs = train_imgs_original
    train_masks = train_masks/255.
    


    train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565
    data_consistency_check(train_imgs,train_masks)
    print(np.min(train_masks))
    #check masks are within 0-1
    assert(np.min(train_masks)==0 and np.max(train_masks)==1)

    print("\ntrain images/masks shape:")
    print(train_imgs.shape)
    print("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
    print("train masks are within 0-1\n")

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train = extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs)
    data_consistency_check(patches_imgs_train, patches_masks_train)

    print("\ntrain PATCHES images/masks shape:")
    print(patches_imgs_train.shape)
    print("train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train#, patches_imgs_test, patches_masks_test


#Load the original data and return the extracted patches for training/testing
def get_data_testing(test_imgs_original, test_groundTruth, Imgs_to_test, patch_height, patch_width):

    test_masks = test_groundTruth
    test_imgs = test_imgs_original
    #test_masks = test_masks/255.

    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border(test_imgs,patch_height,patch_width)
    test_masks = paint_border(test_masks,patch_height,patch_width)

    data_consistency_check(test_imgs, test_masks)
    #check masks are within 0-1
    #assert(np.max(test_masks)==1  and np.min(test_masks)==0)

    print("\ntest images/masks shape:")
    print(test_imgs.shape)
    print("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered(test_imgs,patch_height,patch_width)
    patches_masks_test = extract_ordered(test_masks,patch_height,patch_width)
    data_consistency_check(patches_imgs_test, patches_masks_test)

    print("\ntest PATCHES images/masks shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, patches_masks_test


# Load the original data and return the extracted patches for testing
# return the ground truth in its original shape
def get_data_testing_overlap(test_imgs_original, test_groundTruth, Imgs_to_test, patch_height, patch_width, stride_height, stride_width):
    ### test

    test_masks = test_groundTruth
    test_imgs = test_imgs_original
    #test_masks = test_masks/255.
    #extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test,:,:,:]
    test_masks = test_masks[0:Imgs_to_test,:,:,:]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    #check masks are within 0-1
    #assert(np.max(test_masks)==1  and np.min(test_masks)==0)

    print("\ntest images shape:")
    print(test_imgs.shape)
    print("\ntest mask shape:")
    print(test_masks.shape)
    print("test images range (min-max): " +str(np.min(test_imgs)) +' - '+str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)

    print("\ntest PATCHES images shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3], test_masks


#data consinstency check
def data_consistency_check(imgs,masks):
    assert(len(imgs.shape)==len(masks.shape))
    assert(imgs.shape[0]==masks.shape[0])
    assert(imgs.shape[2]==masks.shape[2])
    assert(imgs.shape[3]==masks.shape[3])
    assert(masks.shape[1]==1)
    assert(imgs.shape[1]==1 or imgs.shape[1]==3)


#extract patches randomly in the full training images
#  -- Inside OR in full image
def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches, inside=True):
    if (N_patches%full_imgs.shape[0] != 0):
        print("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    assert (full_masks.shape[1]==1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    patches = np.empty((N_patches,full_imgs.shape[1],patch_h,patch_w))
    patches_masks = np.empty((N_patches,full_masks.shape[1],patch_h,patch_w))
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches/full_imgs.shape[0])  #N_patches equally divided in the full images
    print("patches per full image: " +str(patch_per_img))
    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        k=0
        while k <patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            # print "y_center " +str(y_center)
            #check whether the patch is fully contained in the FOV
            patch = full_imgs[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_mask = full_masks[i,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot]=patch
            patches_masks[iter_tot]=patch_mask
            iter_tot +=1   #total
            k+=1  #per full_img
    return patches, patches_masks

#Divide all the full_imgs in pacthes
def extract_ordered(full_imgs, patch_h, patch_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    N_patches_h = int(img_h/patch_h) #round to lowest int
    if (img_h%patch_h != 0):
        print("warning: " +str(N_patches_h) +" patches in height, with about " +str(img_h%patch_h) +" pixels left over")
    N_patches_w = int(img_w/patch_w) #round to lowest int
    if (img_h%patch_h != 0):
        print("warning: " +str(N_patches_w) +" patches in width, with about " +str(img_w%patch_w) +" pixels left over")
    print("number of patches per image: " +str(N_patches_h*N_patches_w))
    N_patches_tot = (N_patches_h*N_patches_w)*full_imgs.shape[0]
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))

    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i,:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches  #array with all the full_imgs divided in patches


def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    leftover_h = (img_h-patch_h)%stride_h  #leftover on the h dim
    leftover_w = (img_w-patch_w)%stride_w  #leftover on the w dim
    if (leftover_h != 0):  #change dimension of img_h
        print("\nthe side H is not compatible with the selected stride of " +str(stride_h))
        print("img_h " +str(img_h) + ", patch_h " +str(patch_h) + ", stride_h " +str(stride_h))
        print("(img_h - patch_h) MOD stride_h: " +str(leftover_h))
        print("So the H dim will be padded with additional " +str(stride_h - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_h+(stride_h-leftover_h),img_w))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_h,0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):   #change dimension of img_w
        print("the side W is not compatible with the selected stride of " +str(stride_w))
        print("img_w " +str(img_w) + ", patch_w " +str(patch_w) + ", stride_w " +str(stride_w))
        print("(img_w - patch_w) MOD stride_w: " +str(leftover_w))
        print("So the W dim will be padded with additional " +str(stride_w - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],full_imgs.shape[2],img_w+(stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:full_imgs.shape[2],0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print("new full images shape: \n" +str(full_imgs.shape))
    return full_imgs

#Divide all the full_imgs in pacthes
def extract_ordered_overlap(full_imgs, patch_h, patch_w,stride_h,stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    print("Number of patches on h : " +str(((img_h-patch_h)//stride_h+1)))
    print("Number of patches on w : " +str(((img_w-patch_w)//stride_w+1)))
    print("number of patches per image: " +str(N_patches_img) +", totally for this dataset: " +str(N_patches_tot))
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches  #array with all the full_imgs divided in patches


def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape)==4)  #4D arrays
    assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: " +str(N_patches_h))
    print("N_patches_w: " +str(N_patches_w))
    print("N_patches_img: " +str(N_patches_img))
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
                k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    print(final_avg.shape)
    assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0
    return final_avg


#Recompone the full images with the patches
def recompone(data,N_h,N_w):
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    assert(len(data.shape)==4)
    N_pacth_per_img = N_w*N_h
    assert(data.shape[0]%N_pacth_per_img == 0)
    N_full_imgs = data.shape[0]/N_pacth_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    N_pacth_per_img = N_w*N_h
    N_full_imgs = round(N_full_imgs)
    print(data.shape[1])
    #define and start full recompone
    full_recomp = np.empty((N_full_imgs,data.shape[1],N_h*patch_h,N_w*patch_w))
    k = 0  #iter full img
    s = 0  #iter single patch
    while (s<data.shape[0]):
        #recompone one:
        single_recon = np.empty((data.shape[1],N_h*patch_h,N_w*patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]=data[s]
                s+=1
        full_recomp[k]=single_recon
        k+=1
    assert (k==N_full_imgs)
    return full_recomp


#Extend the full images because patch divison is not exact
def paint_border(data,patch_h,patch_w):
    assert (len(data.shape)==4)  #4D arrays
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    img_h=data.shape[2]
    img_w=data.shape[3]
    new_img_h = 0
    new_img_w = 0
    if (img_h%patch_h)==0:
        new_img_h = img_h
    else:
        new_img_h = ((int(img_h)/int(patch_h))+1)*patch_h
    if (img_w%patch_w)==0:
        new_img_w = img_w
    else:
        new_img_w = ((int(img_w)/int(patch_w))+1)*patch_w
    new_img_w = round(new_img_w)
    new_img_h = round(new_img_h)
    new_data = np.zeros((data.shape[0],data.shape[1],new_img_h,new_img_w))
    new_data[:,:,0:img_h,0:img_w] = data[:,:,:,:]
    return new_data

#return only the pixels contained in the FOV, for both images and masks
def pred_only_FOV(data_imgs,data_masks):
    assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)  #4D arrays
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[2]==data_masks.shape[2])
    assert (data_imgs.shape[3]==data_masks.shape[3])
    assert (data_imgs.shape[1]==1 and data_masks.shape[1]==1)  #check the channel is 1
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
              new_pred_imgs.append(data_imgs[i,:,y,x])
              new_pred_masks.append(data_masks[i,:,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    print(new_pred_imgs.shape)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks

patch_height = 48
patch_width = 48
#number of total patches:
N_subimgs = 19000

#Number of training epochs
N_epochs = 150
batch_size = 32
#if running with nohup
nohup = True

imgs_train.astype(int)
groundTruth_train.astype(int)



patches_imgs_train, patches_masks_train = get_data_training(
    train_imgs_original = imgs_train,
    train_groundTruth = groundTruth_train,  #masks
    patch_height = int(patch_height),
    patch_width = int(patch_width),
    N_subimgs = int(N_subimgs),

)

full_images_to_test = 20

#original test images (for FOV selection)
test_imgs_orig = imgs_test
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
#the border masks provided by the DRIVE

# dimension of the patches

#the stride in case output with average
stride_height = 5
stride_width = 5
assert (stride_height < patch_height and stride_width < patch_width)
#model name

#====== average mode ===========
average_mode = True

img_truth= groundTruth_test

print(groundTruth_test)

patches_imgs_test = None
new_height = None
new_width = None
masks_test  = None
patches_masks_test = None
if average_mode == True:
    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        test_imgs_original =  imgs_test,  #original
        test_groundTruth = groundTruth_test,  #masks
        Imgs_to_test = full_images_to_test,
        patch_height = patch_height,
        patch_width = patch_width,
        stride_height = stride_height,
        stride_width = stride_width
    )
else:
    patches_imgs_test, patches_masks_test = get_data_testing(
        test_imgs_original = imgs_test,  #original
        test_groundTruth = groundTruth_test,  #masks
        Imgs_to_test = full_images_to_test,
        patch_height = patch_height,
        patch_width = patch_width,
    )

print(patches_imgs_test.shape)



""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

print(patches_imgs_test[0])

create_dir('test')

N_sample = min(patches_imgs_train.shape[0],40)


x_train = patches_masks_train
x_test = masks_test
x_train_noisy = patches_imgs_train
x_test_noisy = patches_imgs_test
x_train=x_train.reshape((-1,48,48,1))
#x_test=x_test.reshape((-1,48,48,1))
x_train_noisy = x_train_noisy.reshape((-1,48,48,1))
x_test_noisy = x_test_noisy.reshape((-1,48,48,1))

print(x_train.shape)


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Activation
from tensorflow.keras.models import Model
import warnings; warnings.filterwarnings('ignore')

input_img = Input(shape=(48,48,1), name='Encoder_input')

# Encoder
Enc = Conv2D(16, (3, 3), padding='same', activation='relu', name='Enc_conv2d_1')(input_img)
Enc = MaxPooling2D(pool_size=(2,2), padding='same', name='Enc_max_pooling2d_1')(Enc)
Enc = Conv2D(8,(3, 3), padding='same', activation='relu', name='Enc_conv2d_2')(Enc)
Enc = MaxPooling2D(pool_size=(2,2), padding='same', name='Enc_max_pooling2d_2')(Enc)
Encoded = Conv2D(1, (3, 3), padding='same', activation='sigmoid', name='Enc_conv2d_3')(Enc)

# Instantiate the Encoder Model
encoder = Model(inputs = input_img, outputs = Encoded)
encoder.summary()

# Decoder
Dec = Conv2D(8, (3, 3), padding='same', activation='relu', name ='Dec_conv2d_1')(Encoded)
Dec = UpSampling2D((2, 2), name = 'Dec_upsampling2d_1')(Dec)
Dec = Conv2D(16, (3, 3), padding='same', activation='relu', name ='Dec_conv2d_2')(Dec)
Dec = UpSampling2D((2, 2), name = 'Dec_upsampling2d_2')(Dec)
decoded = Conv2D(1,(3, 3), padding='same', activation='sigmoid', name ='Dec_conv2d_3')(Dec)

# Instantiate the Autoencoder Model
autoencoder = Model(inputs = input_img, outputs = decoded)
autoencoder.summary()

# Compile the autoencoder
autoencoder.compile(loss='mse', optimizer='adam') 

# Train the autoencoder
history = autoencoder.fit(x_train_noisy, x_train, 
                epochs = 100,
                batch_size = 16, 
                shuffle = True, 
                validation_split = 1/10).history

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

tr_loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(tr_loss)+1)



from keras.models import model_from_json

# Save the Encoder
model_json = encoder.to_json()
with open("test/Encoder_model.json", "w") as json_file:
    json_file.write(model_json)
encoder.save_weights("test/Encoder_weights.h5")

# Save the Autoencoder
model_json = autoencoder.to_json()
with open("test/Autoencoder_model.json", "w") as json_file:
    json_file.write(model_json)
autoencoder.save_weights("test/Autoencoder_weights.h5")

from tensorflow.keras.models import model_from_json
import warnings; warnings.filterwarnings('ignore')

with open('test/Encoder_model.json', 'r') as f:
    Myencoder = model_from_json(f.read())
Myencoder.load_weights("test/Encoder_weights.h5")

with open('test/Autoencoder_model.json', 'r') as f:
    MyAutoencoder = model_from_json(f.read())
MyAutoencoder.load_weights("test/Autoencoder_weights.h5")

decoded_imgs = MyAutoencoder.predict(x_test_noisy)
print(decoded_imgs.shape)

# Pick randomly some images from test set
num_images = 1
random_test_images = np.random.randint(x_test.shape[0], size= num_images)

# Predict the Encoder and the Autoencoder outputs from the noisy test images
encoded_imgs = Myencoder.predict(x_test_noisy)
decoded_imgs = MyAutoencoder.predict(x_test_noisy)


print(decoded_imgs.shape)



average_mode = True

decoded_imgs = decoded_imgs.reshape((-1,1,48,48))
pred_imgs = None
orig_imgs = None
gtruth_masks = None
if average_mode == True:
    pred_imgs = recompone_overlap(decoded_imgs, new_height, new_width, stride_height, stride_width)# predictions
    orig_imgs = test_imgs_orig[0:pred_imgs.shape[0],:,:,:]    #originals
    gtruth_masks = masks_test  #ground truth masks
else:
    pred_imgs = recompone(decoded_imgs,13,12)       # predictions
    orig_imgs = recompone(patches_imgs_test,13,12)  # originals
    gtruth_masks = recompone(patches_masks_test,13,12)  #masks

orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]
print("Orig imgs shape: " +str(orig_imgs.shape))
print("pred imgs shape: " +str(pred_imgs.shape))
print("Gtruth imgs shape: " +str(gtruth_masks.shape))


orig_imgs = np.reshape(orig_imgs,(full_images_to_test,height,width))
orig_imgs = orig_imgs.transpose(1,0,2).reshape(height,-1)
print("Orig imgs shape: " +str(orig_imgs.shape))

pred_imgs = np.reshape(pred_imgs,(full_images_to_test,height,width))
pred_imgs = pred_imgs.transpose(1,0,2).reshape(height,-1)
print("Pred imgs shape: " +str(pred_imgs.shape))

gtruth_masks = np.reshape(gtruth_masks,(full_images_to_test,height,width))
gtruth_masks = gtruth_masks.transpose(1,0,2).reshape(height,-1)
print("Gt imgs shape: " +str(gtruth_masks.shape))


np.savetxt("origcdf1.csv", orig_imgs, delimiter=",")
np.savetxt("predcdf1.csv", pred_imgs, delimiter=",")
np.savetxt("gtscdf1.csv", gtruth_masks, delimiter=",")

