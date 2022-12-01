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


def parseData_Abstract():
    """Load Data and Assign Lables"""
    mypath = "data/image/abstract/files"
    imageList = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    dataLabel = []
    dataImage= []
     
    for image in imageList:
        if image == ".DS_Store":
            continue
        currImage=Image.open(mypath + "/" + image) 
        currImage= np.asarray(currImage)
        dataImage.append(currImage) 
    
    lineCount = 0 
    with open('data/image/abstract/abstract_label.csv', newline='') as csvfile:
        csvRead = csv.reader(csvfile)
        for row in csvRead:
            if lineCount == 0: 
                classHeader = row[1:]
                lineCount += 1
                continue
            curr = row[1:]
            curr_label = curr.index(max(curr))
            dataLabel.append(classHeader[curr_label])
            
            
    return dataImage, dataLabel

def parseData_Art():
    """Load Data and Assign Lables"""
    mypath = "data/image/art"
     
    imageList = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    dataLabel = []
    dataImage= []
     
    for image in imageList:
        if image == ".DS_Store":
            continue
        currImage=Image.open(mypath + "/" + image) 
        currImage= np.asarray(currImage)
        dataImage.append(currImage) 
        
        dataLabel.append(image.split("_")[0])

            
    return dataImage, dataLabel
    

def shuffle_data(image_full, label_full, seed=1):
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(np.arange(len(image_full)))
    image_full = image_full[shuffled_index]
    label_full = label_full[shuffled_index]
    return image_full, label_full
