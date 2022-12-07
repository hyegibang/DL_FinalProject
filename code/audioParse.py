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

# ['Happy music', 'Funny music', 'Sad music', 'Tender music', 'Exciting music', 'Angry music', 'Scary music'] --> which is in range [276:283]

def read_labels():
    with open('../data/music/class_labels_indices.csv', encoding='utf-8') as class_map_csv:
        class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
        class_names = class_names[1:]
    class_names = np.array(class_names)
    return class_names


context = {
    'end_time_seconds': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'video_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'start_time_seconds': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    'labels': tf.io.VarLenFeature(tf.int64),    
}

sequence = {
    'audio_embedding': tf.io.FixedLenSequenceFeature([], tf.string, default_value=None ,allow_missing=True)
}


def _parse_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_sequence_example(example_proto, context_features=context, sequence_features=sequence)

def read_audio(): 
    tfrecord_files = '../data/music/bal_train/'
    tfrecord_filenames = os.listdir(tfrecord_files)
    tfrecord_filenames = [(tfrecord_files + "/" + each_fname) 
                      for each_fname 
                      in tfrecord_filenames]
    raw_dataset = tf.data.TFRecordDataset(tfrecord_filenames)
    print(raw_dataset)
    print("Lenght of tfrecord: " + str(len(tfrecord_filenames)))
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset

def parsePerEmotion(parsed_dataset, class_names, desire):
    music_contexts = []
    music_embeddings = []
    music_labels = []
    for i, example in enumerate(parsed_dataset):
        context, sequence = example
        labels = context['labels'].values.numpy()
        if (desire in labels):
            raw_embedding = sequence['audio_embedding'].numpy()
            embedding = tf.io.decode_raw(raw_embedding, tf.int8).numpy()

            if embedding.shape != (10, 128):
                zero_padding = np.zeros((10 - embedding.shape[0], 128), dtype = 'int8')
                embedding = np.concatenate((embedding, zero_padding), axis = 0)

            music_embeddings.append(embedding)
            music_context = (context['video_id'].numpy(), 
                             context['start_time_seconds'].numpy(), 
                             context['end_time_seconds'].numpy())
            music_contexts.append(music_context)
            music_labels.append(class_names[desire].split()[0].lower())
    music_embeddings = np.array(music_embeddings)
    print(i)
    return music_embeddings, music_contexts, music_labels

def getFullEmotion(parsed_dataset, class_names, desired_class): 
    labels, images = np.array([]), np.empty([1,10,128])
    for desire in desired_class: 
        embd, contx, label = parsePerEmotion(parsed_dataset, class_names, desire)
        labels = np.hstack((labels, label))
        images = np.concatenate((images, embd), axis = 0)
    shuf_img, shuf_lab = shuffle_data(images[1:], labels)
    return shuf_img, shuf_lab
   

def visualize(parsed_dataset, class_names): 
    for i,example in enumerate(parsed_dataset):
        context, sequence = example
        raw_embedding = sequence['audio_embedding'].numpy()
        embedding = tf.io.decode_raw(raw_embedding, tf.int8)
        labels = context['labels'].values.numpy()

        plt.title(str(class_names[labels]))
        plt.imshow(embedding,cmap='gray')
        plt.show()
    return 

def shuffle_data(image_full, label_full, seed=1):
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(np.arange(len(image_full)))
    image_full = image_full[shuffled_index]
    label_full = label_full[shuffled_index]
    return image_full, label_full




