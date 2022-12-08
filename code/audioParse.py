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

def read_audio(file_name): 
    tfrecord_files = '../data/music/'+ file_name +'/'
    tfrecord_filenames = os.listdir(tfrecord_files)
    tfrecord_filenames = [(tfrecord_files + "/" + each_fname) 
                      for each_fname 
                      in tfrecord_filenames]
    raw_dataset = tf.data.TFRecordDataset(tfrecord_filenames)
    print("Length of tfrecord: " + str(len(tfrecord_filenames)))
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset

def parsePerEmotion(parsed_dataset, desire):
    music_contexts = []
    music_embeddings = []
#     music_labels = []
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
            if (i%10000) == 0:
                print(f"i = {i}")
        
    print(f"i = {i}")
    music_embeddings = np.array(music_embeddings)

    return music_contexts, music_embeddings  #, music_labels

   
def getFullEmotion(parsed_dataset, desired_class): 
    VAP_mapping = {
        276:(1,0.735,0.772),
        277:(0.918,0.61,0.566),
        278:(0.225,0.333,0.149),
        279:(0.63,0.52,0.509),
        280:(0.95,0.792,0.789),
        281:(0.122,0.83,0.604),
        282:(0.062,0.952,0.528)
    }

    music_context_pd_before_concat = []
    music_embeddings_before_concat = []

    for each_class in desired_class:
        music_contexts, music_embeddings = parsePerEmotion(parsed_dataset, each_class)    
        (music_youtube_ids, music_start_times, music_end_times) = tuple(zip(*music_contexts))

        music_context_pd = pd.DataFrame(data = {"youtube_id": music_youtube_ids, 
                                                "start_time": music_start_times, 
                                                "end_time"  : music_end_times})
        music_context_pd["mood"] = each_class

        valence, arousal, dominance = VAP_mapping[each_class]
        music_context_pd["valence"] = valence
        music_context_pd["arousal"] = arousal
        music_context_pd["dominance"] = dominance

        music_context_pd_before_concat.append(music_context_pd)
        music_embeddings_before_concat.append(music_embeddings)

    music_context_all_moods_pd = pd.concat(music_context_pd_before_concat, 
                                           axis = 0, 
                                           ignore_index = True)

    music_embbedings_all_moods = np.concatenate(music_embeddings_before_concat, 
                                                axis = 0)
    return music_context_all_moods_pd, music_embbedings_all_moods


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

def saveToParquet(file_name, context, embd):
    context_filename = f"{file_name}_music_contexts.parquet"
    embeddings_filename = f"{file_name}_music_embeddings.npy"

    context.to_parquet(context_filename)

    with open(embeddings_filename, "wb") as f:
        np.save(f,embd,allow_pickle = False)

