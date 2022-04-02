from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import os


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") is not None and logs.get("accuracy") > 0.999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


def show_img(idx):
    """idx argument is to decide which img we want to display"""
    current_dir_1 = os.getcwd()

    path_happy = (
        current_dir_1 + "\\Learning\\DeepLearningAI-Intro-Tensorflow\\data\\happy"
    )
    change_dir = os.chdir(path_happy)
    current_dir = os.getcwd()
    list_happy = os.listdir(path_happy)[idx]

    print("Sample happy image:")
    plt.imshow(load_img(f"{os.path.join(current_dir, list_happy)}"))
    plt.show()

    path_sad = current_dir_1 + "\\Learning\\DeepLearningAI-Intro-Tensorflow\\data\\sad"
    change_dir = os.chdir(path_sad)
    current_dir = os.getcwd()
    list_sad = os.listdir(path_sad)[idx]

    print("\nSample sad image:")
    plt.imshow(load_img(f"{os.path.join(current_dir, list_sad)}"))
    plt.show()


def image_generator():
    """this will generate the dataset that we want and soon it will be trained
    # Specify the method to load images from a directory and pass in the appropriate arguments:
    # - directory: should be a relative path to the directory containing the data
    # - targe_size: set this equal to the resolution of each image (excluding the color dimension)
    # - batch_size: number of images the generator yields when asked for a next batch. Set this to 10.
    # - class_mode: How the labels are represented. Should be one of "binary", "categorical" or "sparse".
    """
    current_dir_1 = os.getcwd()
    path = current_dir_1 + "\\Learning\\DeepLearningAI-Intro-Tensorflow\\data"

    # Instantiate the ImageDataGenerator class.
    # Remember to set the rescale argument.
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        directory=path, target_size=(150, 150), batch_size=10, class_mode="binary"
    )

    return train_generator


def train_happy_sad_model(train_generator):
    """
    # Define the model, you can toy around with the architecture.
    # Some helpful tips in case you are stuck:
    # - A good first layer would be a Conv2D layer with an input shape that matches
    #   that of every image in the training set (including the color dimension)
    # - The model will work best with 3 convolutional layers
    # - There should be a Flatten layer in between convolutional and dense layers
    # - The final layer should be a Dense layer with the number of units
    #   and activation function that supports binary classification.

    """
    callbacks = myCallback()

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, (3, 3), activation="relu", input_shape=(150, 150, 3)
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compile the model
    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(learning_rate=0.001),
        metrics=["accuracy"],
    )

    # Train the model
    history = model.fit(x=train_generator, epochs=20, callbacks=[callbacks])
    return history


gen = image_generator()
hist = train_happy_sad_model(gen)
