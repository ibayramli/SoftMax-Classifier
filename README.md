# SoftMax-Classifier
In this repo, I train an image classifer usign cross-entropy loss function. Training this model consists of finding a classifer matrix W that would maximize the log probability of the correct class score for each image. After formulating the cost function and performing backpropagation, I use mini-batch gradient descent to minimize the cross-entropy loss. I then turn to cross-validation to compute the best hyperparameters and save the most optimal classifier. Lastly, I visualize the trained weight vectors in the `softmax_visualized_weights.png`. As seen, each of the trained weigths correponds to a conceptual template of an image class. This agrees with the work mechanism of the linear classifiers, where the predictor function takes the inputted image and computes inner product (a proxy for distance in high-dimensional spaces) with respect to each class template. The template class that is the "closest" to the current image with respect to the standard dot product is then given the highest score. In the final test set, this classifier performs with 33% accuracy.

Note that no external machine learning libraries were used in this exercise. All the computations are performed with help of vectorized array manipulations in numpy. 

To run this code, you must first obtain the CIFAR-10 dataset by first running the following shell code:

wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

tar -xzvf cifar-10-python.tar.gz

rm cifar-10-python.tar.gz

The structure of the documents and data importing tools (data_utils.py) are borrowed from Stanford's CS231n: Convolutional Neural Networks class assignments.
