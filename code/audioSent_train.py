import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import os
from audioSent_model import VAD_audio

def train(model, music_input, VAD_true):
    batch_size = model.batch_size
    input_size = len(music_input) - (len(music_input) %batch_size)
    music_input, VAD_true = shuffle_tfdata(music_input, VAD_true)
    
    lossList = []
    for curr in range(0, input_size, batch_size): 
        currMusicInput = music_input[curr:(curr + batch_size)]
        currVADTrue   = VAD_true[curr:(curr + batch_size)]
        with tf.GradientTape() as tape:
            currVADPred = model(currMusicInput)
            batch_loss = model.loss_function(currVADTrue,currVADPred) 
        grads = tape.gradient(batch_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        lossList.append(batch_loss)
        
    return tf.reduce_mean(np.array(lossList))


def test(model, music_input, VAD_true):
    batch_size = model.batch_size
    input_size = len(music_input) - (len(music_input)%batch_size)
    music_input, VAD_true = shuffle_tfdata(music_input, VAD_true)
    lossList = []
    
    for curr in range(0, input_size, batch_size): 
        currMusicInput = music_input[curr:(curr + batch_size)]
        currVADTrue    = VAD_true[curr:(curr + batch_size)]        
        currVADPred = model(currMusicInput)
        batchLoss = model.loss_function(currVADTrue, currVADPred)
        lossList.append(batchLoss)
    return tf.reduce_mean(np.array(lossList))

def test_VAD(model, music_input,VAD_true): 
    batch_size = model.batch_size
    input_size = len(music_input) - (len(music_input)%batch_size)
    music_input, VAD_true = shuffle_tfdata(music_input, VAD_true)
    VADPred_list = []
    
    for curr in range(0, input_size, batch_size): 
        currMusicInput = music_input[curr:(curr + batch_size)]
        currVADTrue    = VAD_true[curr:(curr + batch_size)]        
        currVADPred = model(currMusicInput)
        VADPred_list.append([currVADTrue, currVADPred])
    return VADPred_list[0]



def getClosest(VAD_pred, music_vad_mapping):
    VAD_pred = np.repeat(VAD_pred, repeats = 7).reshape(3, 7).T
    distances = np.sum(np.square(music_vad_mapping - VAD_pred), axis = 1)
    closest_mood = 276 + np.argmin(distances)
    return closest_mood

def shuffle_tfdata(music_input, VAD_true): 
    input_size = len(music_input)
    shuffled_index = tf.random.shuffle(tf.range(input_size))
    music_input  = tf.gather(music_input, shuffled_index)
    VAD_true = tf.gather(VAD_true, shuffled_index)
    return music_input, VAD_true


def accuracy(model, music_input, emotion_labels, music_vad_mapping):        
    input_size = len(music_input)
    mood_pred_list = []
    for music_index, each_music in enumerate(music_input):
        each_VAD_pred = model(tf.reshape(each_music, [-1, 10, 128, 1]))
        each_closest_mood = getClosest(each_VAD_pred, music_vad_mapping)
        mood_pred_list.append(each_closest_mood)
    mood_pred_array = np.array(mood_pred_list)

    mood_accuracy_array = np.array([mood_pred_array == emotion_labels])
    return np.mean(mood_accuracy_array)

def train_full(model, numEpoch, audio_train, label_train):
    for epoch in range(numEpoch):
        print("Epoch: " + str(epoch + 1))
        inputTrainNorm = (
            audio_train + tf.random.normal(audio_train.shape, 
                                                  mean = 0, 
                                                  stddev = 0.05, 
                                                  dtype = tf.float32))
    
        train_loss = train(model, inputTrainNorm, label_train)
        print(f"train_loss = {train_loss}", "\n")    
    