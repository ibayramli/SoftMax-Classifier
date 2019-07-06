import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]
  # Get the classified scores matrix. (N, C)
  score_matrix = X.dot(W) 
  X_trans = X.transpose()
  
  # Exponentiate and normalize
  exp_score_matrix = np.exp(score_matrix) # (N, C)
  softmax_score_matrix = exp_score_matrix/np.sum(exp_score_matrix, axis=1)[:, None] # (N, C)
  
  # Get the correct class score for each image 
  correct_class_vector = softmax_score_matrix[np.arange(num_train), y] # (N, 1)

  # Compute the loss for each class
  loss_vector = -np.log(correct_class_vector)
  loss = np.sum(loss_vector)
  
  # Take the average loss
  loss /= num_train

  # Regularize
  loss += 0.5 * reg * np.sum(W**2)

  # Carry out the backprop with the naive algorithm en masse
  # Here we will utilize the following algebraic trick which makes backprop easier:
  # -log(correct_class_score_exp/sum_class_scores_exp) = -log(correct_class_score) + log(sum_class_scores_exp)

  dlog = 1/np.sum(exp_score_matrix, axis=1) # (N, 1)
  dexp = exp_score_matrix*dlog[:, None] # (N, C)
  
  # Multiply x_i by dexp_j, j \in range(0, C) and put it in the jth place of dW
  dW = X_trans.dot(dexp) # (D, C)

  binary_matrix = np.zeros((num_train, num_classes))
  binary_matrix[np.arange(num_train), y] = 1
  
  dW -= X_trans.dot(binary_matrix)
  dW /= num_train
  dW += reg*W

  return loss, dW

