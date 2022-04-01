# Course-1

# Week-1 (Housing Prices)
In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.

Imagine that house pricing is as easy as:
A house has a base cost of 50k, and every additional bedroom adds a cost of 50k. This will make a 1 bedroom house cost 100k, a 2 bedroom house cost 150k etc.


# Week-2 (Implementing Callbacks in TensorFlow)
In the course you learned how to do classification using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy. In the lecture you saw how this was done for the loss but here you will be using accuracy instead.

Some notes:

Given the architecture of the net, it should succeed in less than 10 epochs.
When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!" and stop training.
If you add any additional variables, make sure you use the same names as the ones used in the class. This is important for the function signatures (the parameters and names) of the callbacks.

# Week-3 (Improve MNIST with Convolutions)
In the videos you looked at how you would improve Fashion MNIST using Convolutions. For this exercise see if you can improve MNIST to 99.5% accuracy or more by adding only a single convolutional layer and a single MaxPooling 2D layer to the model from the assignment of the previous week.

You should stop training once the accuracy goes above this amount. It should happen in less than 10 epochs, so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your callback.

When 99.5% accuracy has been hit, you should print out the string "Reached 99.5% accuracy so cancelling training!"

# Week-4 (Handling Complex Images)
In this assignment you will be using the happy or sad dataset, which contains 80 images of emoji-like faces, 40 happy and 40 sad.

Create a convolutional neural network that trains to 100% accuracy on these images, which cancels training upon hitting training accuracy of >.999