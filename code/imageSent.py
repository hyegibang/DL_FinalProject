import tensorflow as tf

class ImageSentModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv2D(
            filters = 8, 
            kernel_size = 3, 
            padding = "SAME", 
            activation = tf.keras.layers.LeakyReLU())
        self.conv2 = tf.keras.layers.Conv2D(
            filters = 16, 
            kernel_size = 3, 
            padding = "SAME",
            activation = tf.keras.layers.LeakyReLU())
        self.conv3 = tf.keras.layers.Conv2D(
            filters = 32, 
            kernel_size = 3, 
            padding = "SAME", 
            activation = tf.keras.layers.LeakyReLU())
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_layer = tf.keras.layers.Dense(60, activation = tf.keras.layers.LeakyReLU() )
        self.output_layer = tf.keras.layers.Dense(3, activation = "sigmoid") 

        self.loss_tracker = tf.keras.metrics.Mean(name="mse_loss")
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.acc_tracker = tf.keras.metrics.Mean(name="acc")

    def call(self, inputs):
        images, _ = inputs
        # res = self.conv(images)
        # res = self.conv(res + images)
        # res = self.conv(res + images)
        # res = self.conv(res + images)
        res = self.conv1(images)
        res = self.conv2(res)
        res = self.conv3(res)
        res = self.maxpool(res)
        res = self.flatten(res)
        res = self.hidden_layer(res)
        res = self.output_layer(res)
        return res

    def compile(self, optimizer, VAD_map):
        super().compile()
        self.optimizer = optimizer
        self.VAD_map = VAD_map

    @tf.autograph.experimental.do_not_convert
    def getClosest(self, VAD_vals):
        closest_moods = tf.map_fn(
            fn=lambda t: tf.math.argmin(tf.math.reduce_sum(tf.math.square(self.VAD_map - t), axis=1)), 
            elems=VAD_vals, 
            fn_output_signature=tf.int64)
        return closest_moods
    
    def acc_fn(self, vad_true, vad_pred):
        labels_true = self.getClosest(vad_true)
        # print(vad_pred)
        labels_pred = self.getClosest(vad_pred)
        return tf.math.reduce_mean(tf.keras.metrics.binary_accuracy(labels_true, labels_pred))
    
    def batch_step(self, data, training=False):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x)
            # Compute the loss value (the loss function is configured in `compile()`)
            loss = self.mse_loss(y, y_pred)

        acc = self.acc_fn(y, y_pred)

        if training:
            # Compute gradients
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return self.batch_step(data, training=True)

    def test_step(self, data):
        return self.batch_step(data, training=False)

    def predict_step(self, data):
        print(data.shape)
        return self.batch_step(data, training=False)

