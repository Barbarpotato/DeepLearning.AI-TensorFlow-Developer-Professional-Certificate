import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        """Define the correct function signature for on_epoch_end
        Stop training once the above condition is met
        """
        if logs.get("accuracy") is not None and logs.get("accuracy") > 0.97:
            print("\nReached 93% accuracy so cancelling training!")
            self.model.stop_training = True


def get_data():
    """Get current working directory,
    Append data/mnist.npz to the previous path to get the full path"""
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, "mnist.npz")
    return data_path


def train_mnist(x_train, y_train, index_img):
    """First Instantiate the callback class,
    Then define the model that should have 3 layers:
    - A Flatten layer that receives inputs with the same shape as the images
    - A Dense layer with 512 units and ReLU activation function
    - A Dense layer with 10 units and softmax activation function,
    Create Compile model
    Create Training model"""

    callbacks = myCallback()

    model = tf.keras.models.Sequential(
        [
            keras.layers.Flatten(),
            keras.layers.Dense(units=512, activation=tf.nn.relu),
            keras.layers.Dense(units=10, activation=tf.nn.softmax),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Fit the model for 10 epochs adding the callbacks
    # and save the training history
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

    ### END CODE HERE
    predict = model.predict(x_test)
    print(predict[index_img])
    print(y_test[index_img])
    plt.imshow(x_test[index_img])
    plt.show()


data_path = get_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=data_path)
x_train = x_train / 255.0
data_shape = x_train.shape
print(
    f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})"
)


hist = train_mnist(x_train, y_train, 1)
