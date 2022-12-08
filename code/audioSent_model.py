import tensorflow as tf


class VAD_audio(tf.keras.Model):
    def __init__(self):
        super().__init__()
       
        self.music_length = 10
        self.VGG_size = 128 
        self.batch_size = 128 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.003) 

        self.conv1 = tf.keras.layers.Conv2D(filters = 8, kernel_size = [5, 16], strides = (1, 2), padding = "same", activation = tf.keras.layers.LeakyReLU())
        self.conv2 = tf.keras.layers.Conv2D(filters = 8, kernel_size = [5, 16], strides = (2, 2), padding = "same", activation = tf.keras.layers.LeakyReLU())
        self.conv3 = tf.keras.layers.Conv2D(filters = 8, kernel_size = [5, 16], strides = (2, 2), padding = "same", activation = tf.keras.layers.LeakyReLU())
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation = tf.keras.layers.LeakyReLU() )
        self.dense2 = tf.keras.layers.Dense(3,activation = "sigmoid") 
        
    def call(self, inputAudio):
        layer = self.conv1(inputAudio)
        layer = self.conv2(layer)
        layer = self.conv3(layer)
        layer = self.flatten(layer)
        layer = self.dense1(layer)
        layer = self.dense2(layer)
        return layer

    def loss_function(self, VAD_true, VAD_pred):
        diff = tf.math.square(VAD_true - VAD_pred)
        squared_distances = tf.reduce_sum(diff, axis = 1)
        loss = tf.reduce_mean(squared_distances)
        return loss
