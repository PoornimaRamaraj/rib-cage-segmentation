# coding: utf-8

# In[1]:


# Import all packages
#--------------------------------------------------------------------------------------
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import random
import subprocess
import shlex
import math
# import utils
import pprint
import nrrd

import numpy as np
import pandas as pd
import skimage.io as io
import skimage.transform as trans

import cv2
import pickle
from parse import parse

from skimage.transform import resize
from tempfile import TemporaryFile
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib

import tensorflow as tf
from tensorflow.python.client import device_lib

import keras
from keras import backend as K
from keras import regularizers

from keras.models import *
from keras.models import Sequential, load_model
from keras.models import Model, model_from_json

from keras.layers import *
from keras.layers import Input, Embedding
from keras.layers import Dense, Dropout, Reshape, Flatten
from keras.layers import Conv2D, Conv1D, Convolution2D
from keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D, BatchNormalization
from keras.layers import merge

from keras.preprocessing import image

from keras.optimizers import SGD
from keras.optimizers import *

from keras.utils import multi_gpu_model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, Callback
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

print(K.tensorflow_backend._get_available_gpus())


# In[2]:


# Setting empty list for train and test data 
# x = train data, y = label data 
#---------------------------------------------------------------------------------------
train_x = []
train_y = []

val_x=[]
val_y=[]

test_x = []
test_y = []


# In[3]:


# Load path directory of data
#-----------------------------------------------------------------------------------
base_seg_dir = '/media/samba_share/poornima/XrayLungSeg/data/labels/seg_files/png/'


# In[4]:


# def read_text()
with open("/media/samba_share/poornima/XrayLungSeg/data/labels/seg_files/split/train.txt", "r") as f:
    data = f.readlines()
    for line in data:
        patient_name =line.strip().split()[0]
        cxr = os.path.join(base_seg_dir,patient_name,"cxr",patient_name)
        cxr = "{}.png".format(cxr)  
        print cxr
        cxr = cv2.imread(cxr,0)    
        cxr = cv2.resize(cxr,(256,256),cv2.INTER_CUBIC)
        plt.imshow(cxr,cmap="gray")
        plt.show()
       
        seg = os.path.join(base_seg_dir,patient_name,"seg",patient_name)
        seg = "{}.png".format(seg)
        seg = cv2.imread(seg,0)
        seg = cv2.resize(seg,(256,256),cv2.INTER_CUBIC) 
        print np.max(seg)
        seg = seg/255.0
        print np.max(seg)   
        plt.imshow(seg,cmap="gray")
        plt.show()
        if cxr.shape == seg.shape:
    #        
            train_input = np.expand_dims(cxr, axis=-1)
            train_x.append(train_input)
            
            train_output = np.expand_dims(seg, axis=-1)
            train_y.append(train_output)


# In[5]:


with open("/media/samba_share/poornima/XrayLungSeg/data/labels/seg_files/split/val.txt", "r") as f:
    data = f.readlines()
    for line in data:
        patient_name =line.strip().split()[0]
        cxr = os.path.join(base_seg_dir,patient_name,"cxr",patient_name)
        cxr = "{}.png".format(cxr) 
        print cxr
        cxr = cv2.imread(cxr,0) 
        cxr = cv2.resize(cxr,(256,256),cv2.INTER_CUBIC)
        plt.imshow(cxr,cmap="gray")
        plt.show()
       
        seg = os.path.join(base_seg_dir,patient_name,"seg",patient_name)
        seg = "{}.png".format(seg)
        seg = cv2.imread(seg,0)
        seg = cv2.resize(seg,(256,256),cv2.INTER_CUBIC)        
        seg = seg/255.0
        print np.max(seg)   
        plt.imshow(seg,cmap="gray")
        plt.show()
        if cxr.shape == seg.shape:
            print "CT data and Segmentation data are of equal size"
    #        
            val_input = np.expand_dims(cxr, axis=-1)
            val_x.append(val_input)
                    # Segmentation data

            val_output = np.expand_dims(seg, axis=-1)
            val_y.append(val_output)



        else:
                print "Cxr data and Segmentation data are NOT of equal size"


# In[6]:


# # Loading data - test 
# # ------------------------------------------------------------------------------------------
base_dir_test=("/media/samba_share/poornima/XrayLungSeg/data/cxr/02_China_Mont_1024/China_Mont/")
with open("/media/samba_share/poornima/XrayLungSeg/data/labels/seg_files/split/test.txt", "r") as f:
    data = f.readlines()
    for line in data:
        print line
        patient_name =line.strip().split()[0]
        cxr = os.path.join(base_dir_test,patient_name)
       
        cxr = "{}.png".format(cxr)  
        print cxr
        cxr = cv2.imread(cxr,0) 
        cxr = cv2.resize(cxr,(256,256),cv2.INTER_CUBIC)
        plt.imshow(cxr,cmap="gray")
        plt.show()
        
        test_input = np.expand_dims(cxr, axis=-1)
        test_x.append(test_input)
        

        


# In[7]:


# Defining the Model- 2D U-Net 
#----------------------------------------------------------------------------------

def unet_relu_example(inputs, pretrained_weights=None):
    
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv1_1')(inputs)
    conv1 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv1_1")(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv1_2')(conv1)
    conv1 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv1_2")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv2_1')(pool1)
    conv2 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv2_1")(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv2_2')(conv2)
    conv2 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv2_2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv3_1')(pool2)
    conv3 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv3_1")(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv3_2')(conv3)
    conv3 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv3_2")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv4_1')(pool3)
    conv4 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv4_1")(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv4_2')(conv4)
    conv4 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv4_2")(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv5_1')(pool4)
    conv5 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv5_1")(conv5)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv5_2')(conv5)
    conv5 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv5_2")(conv5)
    drop5 = Dropout(0.5)(conv5)
    
 
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv6_1')(merge6)
    conv6 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv6_1")(conv6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv6_2')(conv6)
    conv6 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv6_2")(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3,up7],axis = 3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv7_1')(merge7)
    conv7 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv7_1")(conv7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv7_2')(conv7)
    conv7 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv7_2")(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2,up8],axis = 3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv8_1')(merge8)
    conv8 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv8_1")(conv8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal' , name='conv8_2')(conv8)
    conv8 = BatchNormalization(axis=-1, epsilon=1e-5, name="bn_conv8_2")(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    output = Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    model = Model(inputs=inputs, outputs=output)
    
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


# In[8]:


# Setting data and hyperparameters
#------------------------------------------------------------------------------------
# Set training data
imgs_x_train = np.array(train_x)
imgs_y_train = np.array(train_y)
print("Shape of Training data         :"+ str(imgs_x_train.shape))
print("Shape of Training label data   :"+ str(imgs_y_train.shape))

# Set validation data
imgs_x_val = np.array(val_x)
imgs_y_val = np.array(val_y)
validation_data = (imgs_x_val,imgs_y_val)
print("Shape of val data          :"+ str(imgs_x_val.shape))
print("Shape of val label data    :"+ str(imgs_y_val.shape))

# set test data
imgs_x_test = np.array(test_x)
print ("Shape of Testing data          :"+ str(imgs_x_test.shape))

# Set save model path
model_save_dir = '/media/samba_share/poornima/XrayLungSeg/models/'
date_version = "11-06-2018"
model_save_path = model_save_dir + date_version + '_unet_weights-{epoch:02d}-{val_loss:.02f}.hdf5'

# Set the hyperparameter
batch_size = 4
epochs = 100
learning_rate = 1e-4
op = keras.optimizers.Adam(lr=learning_rate) ## Adagrad, SGD, RMSprop.... , 

# Save check point
checkpointer = ModelCheckpoint(filepath=model_save_path, verbose=1 , save_weights_only=False, save_best_only=False)
callbacks = [checkpointer]

# Set input to be given to the model
input_shape = (imgs_x_train.shape[1], imgs_x_train.shape[2], imgs_x_train.shape[3])
print("Input shape                    :" + str(input_shape)) 
x_input = keras.layers.Input(input_shape)


# In[9]:


# Set the model
model = unet_relu_example(x_input)
print model.summary()


# In[10]:


# Configure the model for training
#--------------------------------------------------------------------------------------------
model.compile(optimizer=op, loss='binary_crossentropy', metrics=[]) 


# In[11]:


# Trains the model for a given number of epochs (iterations on a dataset)
#--------------------------------------------------------------------------------------------
history = model.fit(imgs_x_train, imgs_y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data = validation_data,
                    callbacks=callbacks)


# In[12]:


# Loading weights from best model
#----------------------------------------------------------------------------------------------------
model.load_weights('/media/samba_share/poornima/XrayLungSeg/models/11-06-2018_unet_weights-15-0.10.hdf5')


# In[13]:


# Get organ result 
#-----------------------------------------------------------------------------------------------
def get_organ_result(binary_result,slice_id, organ_id):
    sample_slice = binary_result[slice_id]
    sample_slice = sample_slice[:,:,organ_id]
    print ("Slice ID is:" + str (slice_id))      
#     plt.imshow(sample_slice,cmap="gray")
#     plt.show()
    return sample_slice

#  Dice score 
#---------------------------------------------------------------------------------------
def diceCoefficient(segmentationTrue, segmentationResult):
    overlap = np.logical_and(segmentationTrue, segmentationResult)
    dice = np.sum(overlap)*2 / (np.sum(segmentationTrue)+ np.sum(segmentationResult))
    return dice

def get_convex_lung(img):
    """
        Postprocessing of a segmented lung region.
        Parameter:
            img = prediction of a segmentation model
        Return :
            convex hull of the largest island of prediction.
    """
    
    _,thresh = cv2.threshold(img.copy(),0,255,cv2.THRESH_BINARY)

    # find contours
   
    _,contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # sort contours to the largest one is first
   
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
   
    # choose the largest one
   
    if not contours:
        return None, None
    
    ## compute area and collect all contours lager than 50000 pixels.
    contours = [ x for x in contours if cv2.contourArea(x) > 15000]
  

    points = contours[0]
    length = contours[0].shape[0]
    for cnt in contours:
        points = np.append(points, cnt)
        n, _, _ = cnt.shape
        length += n

    points = points.reshape(length, 1, 2)  # list of contours -> a contour with dots. [(id, x, y)]

    ## Make convex
    points = cv2.convexHull(points)
    convex_hull = np.zeros(img.shape, np.uint8)
    convex_hull2 = cv2.fillPoly(convex_hull, pts=[points], color=255)
    
  

    return convex_hull2

# Setting empty list for calculating dice score 
#--------------------------------------------------------------------------------------------------
lung_dice_score = []


# In[14]:


# Prediction of val set
#----------------------------------------------------------------------------------------------------
result_prediction = model.predict(imgs_x_val, batch_size=batch_size, verbose=1, steps=None)
print ("Shape of result of Prediction:" + str(result_prediction.shape))
# saveResult('/media/samba_share/data/organdose/MODELS/Results',result_prediction)
# Setting Threshold
#---------------------------------------------------------------------------------------------------
indices = result_prediction > 0.5
print indices.shape
binary_result = np.zeros(result_prediction.shape)
binary_result[indices] = 1


# In[15]:


with open("/media/samba_share/poornima/XrayLungSeg/data/labels/seg_files/split/val.txt", "r") as f:
    i=0
    data = f.readlines()
    for line in data:
        print line
        print i
        patient_name =line.strip().split()[0]
        cxr = os.path.join(base_seg_dir,patient_name,"cxr",patient_name)
        cxr = "{}.png".format(cxr)  
        cxr = cv2.imread(cxr,0)    
        cxr = cv2.resize(cxr,(256,256),cv2.INTER_CUBIC)
        plt.imshow(cxr,cmap="gray")
        plt.show()
       
        seg = os.path.join(base_seg_dir,patient_name,"seg",patient_name)
        seg = "{}.png".format(seg)
        seg = cv2.imread(seg,0)
        seg = cv2.resize(seg,(256,256),cv2.INTER_CUBIC)        
        seg = seg/255.0
        np_max = np.max(seg)  
        print "original"
        plt.imshow(seg,cmap="gray")
        plt.show()
    
    
#         true_lung = seg
#         plt.imshow(true_lung,cmap="gray")
#         plt.show()
#         print ("Ground truth of Lungs{}".format(np.sum(true_lung)))
#         if np.sum(true_lung) == 0:
#             continue
#         print " np"
#         plt.imshow(true_lung,cmap="gray")
#         plt.show()
#         print true_lung.shape
        

        print "Prediction results of lungs: "
        organ_ID_Lungs = 0
        predict_lung = get_organ_result(binary_result,i, organ_ID_Lungs)
        print "predict_lung"
        plt.imshow(predict_lung,cmap= "gray")
        plt.show()
        img= predict_lung
#         print "img"
#         plt.imshow(img,cmap= "gray")
#         plt.show()
        img_1=img.copy()
#         print "img_1"
#         plt.imshow(img_1,cmap= "gray")
#         plt.show()
        img_1[img!=1]=0
        img_1=get_convex_lung(img_1.astype(np.uint8))
        img_1=img_1/255.0
        print "aftr post"
        plt.imshow(img_1,cmap= "gray")
        plt.show()
        print np.max(img_1)
#         cxr[predict_lung==0] = 0
#         plt.imshow(cxr,cmap= "gray")
#         plt.show()
    #     putting into rgb channels

        
#         bgr_image = np.zeros((256, 256, 3), dtype=np.uint8)
#         bgr_image[:,:,0] = img # b
#         bgr_image[:,:,1] = img # g
#         bgr_image[:,:,2] = img # r
#         bgr_image = bgr_image.astype(dtype=np.float32) # convert data type to float32
#         print bgr_image.shape
#         plt.imshow(bgr_image)
#         plt.show()
#         bgr_image=np.rot90(bgr_image)


#         plt.imshow(bgr_image)
#         plt.show()
        
#         cxr[bgr_image==0] = 0
#         plt.imshow(cxr,cmap= "gray")
#         plt.show()

    #     image = image[:,:,::-1] # RGB -> BGR
    #     bgr_image = bgr_image - np.array((104.00698793,116.66876762,122.67891434)) # Mean subtraction
    #     plt.imshow(bgr_image)
    #     plt.show()
    # #     end
        lung_dice = diceCoefficient(seg, img_1)
        print("Lung dice{}".format(lung_dice))
        lung_dice_score.append(lung_dice)
        i+=1
    np_lung = np.array(lung_dice_score)
    mean_dice_lung = np.mean(np_lung)
    print("Mean Dice score of Lungs           :"+str(mean_dice_lung))


# In[16]:


# prediction of test set 
result_prediction = model.predict(imgs_x_test, batch_size=batch_size, verbose=1, steps=None)
print ("Shape of result of Prediction:" + str(result_prediction.shape))
# saveResult('/media/samba_share/data/organdose/MODELS/Results',result_prediction)
# Setting Threshold
#---------------------------------------------------------------------------------------------------
indices = result_prediction > 0.5
print indices.shape
binary_result = np.zeros(result_prediction.shape)
binary_result[indices] = 1


# In[17]:


def make_dir(target_dir):
    if not os.path.exists(target_dir):
        cmd = "sudo mkdir -p {}".format(target_dir)
        os.system(cmd)
        cmd = 'sudo chmod 777 -R {}'.format(target_dir)
        os.system(cmd)


# In[18]:


# make sure image in rgb channel
# make prediction on test tests
base_dir_test=("/media/samba_share/poornima/XrayLungSeg/data/cxr/02_China_Mont_1024/China_Mont/")
with open("/media/samba_share/poornima/XrayLungSeg/data/labels/seg_files/split/test.txt", "r") as f:
    i=0
    data = f.readlines()
    for line in data:
        print line
        patient_name =line.strip().split()[0]
        cxr = os.path.join(base_dir_test,patient_name)
       
        cxr = "{}.png".format(cxr)  
        print cxr
        cxr = cv2.imread(cxr) 
        cxr = cv2.resize(cxr,(256,256),cv2.INTER_CUBIC)
        
        true_lung = cxr
        print ("Ground truth of Lungs{}".format(np.sum(true_lung)))
        if np.sum(true_lung) == 0:
            continue
        plt.imshow(true_lung,cmap="gray")
        plt.show()
        print true_lung.shape
        true_lung=np.rot90(true_lung)
#         make_dir("/media/samba_share/data/XrayLungSeg/data/cxr/Prediction_masks/{}".format(patient_name))
#         nrrd.write("/media/samba_share/data/XrayLungSeg/data/cxr/Prediction_masks/{}/{}.nrrd".format(patient_name,patient_name),true_lung)

        print "Prediction results of lungs: "
        organ_ID_Lungs = 0
        predict_lung = get_organ_result(binary_result,i, organ_ID_Lungs)
        plt.imshow(predict_lung)
        plt.show()
        img= predict_lung
        img_1=img.copy()
        img_1[img!=1]=0
        img_1=get_convex_lung(img_1.astype(np.uint8))
        print "after post process"
#         plt.imshow(img_1,cmap="gray")
#         plt.show()

#     putting into rgb channels
    
#     print type(img_1)
#     img_1 = img_1.astype(dtype=np.float32)
#     plt.imshow(img_1)
#     plt.show()
    

        bgr_image = np.zeros((256, 256, 3), dtype=np.uint8)
        bgr_image[:,:,0] = img_1 # b
        bgr_image[:,:,1] = img_1 # g
        bgr_image[:,:,2] = img_1 # r
        bgr_image = bgr_image.astype(dtype=np.float32) 
        bgr_image=np.rot90(bgr_image)
        bgr_image = bgr_image/255.0
#         plt.imshow(bgr_image)
#         plt.show()
       
#         bgr_image=np.rot90(bgr_image)
        true_lung[bgr_image==0] = 0
        true_lung=np.rot90(true_lung)
        true_lung=np.rot90(true_lung)
        true_lung=np.rot90(true_lung)
        plt.imshow(true_lung,cmap= "gray")
        plt.show()
        i+= 1
        
#         nrrd.write("/media/samba_share/data/XrayLungSeg/data/cxr/Prediction_masks/{}/{}.nrrd".format(patient_name,i),bgr_image)
    
    # convert data type to float32
#     img= bgr_image.argmax(axis=0)
#     img_1 = img.copy()
#     img_1[img!=1]=0
#     img_1=get_convex_lung(bgr_image.astype(np.uint8))
    
#     print bgr_image.shape
#     plt.imshow(bgr_image)
#     plt.show()
#     bgr_image=np.rot90(bgr_image)
   
#     nrrd.write("/media/samba_share/data/XrayLungSeg/data/cxr/seg/{}.nrrd".format(patient_name),bgr_image)
#     plt.imshow(bgr_image)
#     plt.show()
   
#       true_lung[bgr_image==0] = 0
#       plt.imshow(true_lung,cmap= "gray")
#       plt.show()


# In[ ]:




