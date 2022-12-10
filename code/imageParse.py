from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import csv
import numpy as np
import IPython.display as display
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from PIL import Image
# from skimage.transform import resize
import cv2


def input_prep_fn(x):
    ## TODO: Perform preprocessing on the data as appropriate
    # out = np.expand_dims(x, axis = -1)
    out = x.astype('float32') / 255.0
    out = tf.image.resize(out, [32,32])
    return out


def parseData_Art():
    """Load Data and Assign Lables"""
    mypath = "../data/image/art"
     
    imageList = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    dataLabel = np.array([])
#     dataImage= np.empty([255,255,3])
    dataImage = []
     
    for image in imageList:
        if image == ".DS_Store":
            continue
        currImage = cv2.imread(mypath + "/" + image)
        currImage = cv2.resize(currImage, dsize=(255,255))
        dataImage.append(currImage)
        label = image.split("_")[0]
        
        if label == "amusement": 
            label = "funny"
        elif label == "contentment" : # content for abstract 
            label = "happy"
        elif label == "anger": 
            label = "angry"
        elif label == "excitement": 
            label = "exciting"
        elif label == "fear": 
            label = "scary"
        elif label == "disgust" or label == "awe": 
            label = "tender"
        dataLabel = np.hstack((dataLabel, label))
        
    dataImage = np.array(dataImage)
    shuf_img, shuf_lab = shuffle_data(dataImage[1:], dataLabel)
    train_test = int(len(shuf_img) * 0.8)
    return shuf_img[:train_test], shuf_lab[:train_test], shuf_img[train_test:], shuf_lab[train_test:]
    

def shuffle_data(image_full, label_full, seed=1):
#     print(image_full)
    image_full = np.array(image_full)
    label_full = np.array(label_full)
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(np.arange(len(image_full)))
    image_full = image_full[shuffled_index]
    label_full = label_full[shuffled_index]
    return image_full, label_full

def parseData_Abs():
    """Load Data and Assign Lables"""
    mypath = "../data/image/abstract/files"
    imageList = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    dataLabel = []
    dataImage= []
     
    for image in imageList:
        if image == ".DS_Store":
            continue
        currImage = cv2.imread(mypath + "/" + image)
        currImage = cv2.resize(currImage, dsize=(255,255))
        dataImage.append(currImage)
    lineCount = 0 
    with open('../data/image/abstract/abstract_label.csv', newline='') as csvfile:
        csvRead = csv.reader(csvfile)
        for row in csvRead:
            if lineCount == 0: 
                classHeader = row[1:]
                lineCount += 1
                continue
            curr = row[1:]
            curr_label = curr.index(max(curr))
            label = classHeader[curr_label]
            if label == "'Amusement'": 
                label = "funny"
            elif label == "'Content'" :
                label = "happy"
            elif label == "'Anger'": 
                label = "angry"
            elif label == "'Excitement'": 
                label = "exciting"
            elif label == "'Fear'": 
                label = "scary"
            elif label == "'Disgust'" or label == "'Awe'": 
                label = "tender"
            elif label == "'Sad'": 
                label = "sad"
            dataLabel.append(label)
        dataImage = np.array(dataImage)
    shuf_img, shuf_lab = shuffle_data(dataImage, dataLabel)
    train_test = int(len(shuf_img) * 0.8)
    return shuf_img[:train_test], shuf_lab[:train_test], shuf_img[train_test:], shuf_lab[train_test:]
