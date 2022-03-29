import tensorflow as tf
import numpy as np
from tensorflow import keras


def house_model():
    """In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.
    imagine that house pricing is as easy as:
    A house has a base cost of 50k, and every additional bedroom adds a cost of 50k.
    This will make a 1 bedroom house cost 100k, a 2 bedroom house cost 150k etc.
    How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

    Hint: Your network might work better if you scale the house price down.
    You don't have to give the answer 400...
    it might be better to create something that predicts the number 4,
    and then your answer is in the 'hundreds of thousands' etc.
    """

    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(xs, ys, epochs=1000)
    return model


# Get Your Trained Model
model = house_model()
new_y = 7.0
prediction = model.predict([new_y])[0]
print(prediction)
