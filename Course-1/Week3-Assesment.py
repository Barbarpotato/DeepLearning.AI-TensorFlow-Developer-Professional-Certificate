import os
import tensorflow as tf
import numpy as np


class myCallback(tf.keras.callbacks.Callback):
    # Define the method that checks the accuracy at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        """checks the accuracy of our model
        if accuracy same or more than 99.5%, than stop the training
        otherwise print the information accuracy model is not reached 99.5%
        after the training is done
        """
        if logs.get("accuracy") is not None and logs.get("accuracy") >= 0.995:
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True


def get_data():
    """Get current working directory,
    Append data/mnist.npz to the previous path to get the full path"""
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, "mnist.npz")
    return data_path


def reshape_and_normalize(images):
    """Reshape the images to add an extra dimension
    then Normalize the pixel values
    """
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    images = images / 255
    return images


def convolutional_model():
    """Define the model, that have have 5 layers:
    # - A Conv2D layer with 32 filters, a kernel_size of 3x3, ReLU activation function
    #    and an input shape that matches that of every image in the training set
    # - A MaxPooling2D layer with a pool_size of 2x2
    # - A Flatten layer with no arguments
    # - A Dense layer with 128 units and ReLU activation function
    # - A Dense layer with 10 units and softmax activation function
    """
    model = tf.keras.models.Sequential(
        [
            # Add convolutions and max pooling
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Add the same layers as before
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


# Load the datasets
data_path = get_data()
(training_images, training_labels), (
    test_images,
    test_labels,
) = tf.keras.datasets.mnist.load_data(path=data_path)

# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

# Apply your function
training_images = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")

# Save your untrained model
model = convolutional_model()

# Instantiate the callback class
callbacks = myCallback()

# Train your model (this can take up to 5 minutes)
history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
