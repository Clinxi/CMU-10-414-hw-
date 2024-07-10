"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, 'rb') as f:
      image_content = f.read()
      # image_content[4:8][0] indicates the number of images in the file
      # here ">I" means big-end and unsigned int
      image_num = struct.unpack('>I', image_content[4:8])[0]
      X = np.array(struct.unpack('B'*784*image_num, \
                                      image_content[16: 16 + image_num*784])\
                                      , dtype = np.float32)
      X.resize((image_num, 784))
      X /= 255.0

    with gzip.open(label_filename, 'rb') as f:
      label_content = f.read()
      label_num = struct.unpack('>I', label_content[4:8])[0]
      y = np.array([struct.unpack('B', label_content[8+i: 9+i])[0] for i in range(label_num)]\
                                    , dtype = np.uint8)
    return X, y
    # raise NotImplementedError()
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size = Z.shape[0]
    Z_sum = ndl.summation(ndl.log(ndl.summation(ndl.exp(Z), axes=(1, ))))
    Z_y = ndl.summation(Z * y_one_hot) # element-wise
    return (Z_sum - Z_y)/batch_size
    # raise NotImplementedError()
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    # X:num * input_dim. 
    # W1:input_dim * hidden_dim. 
    # W2:hidden_dim * classes
    batch_num = int(X.shape[0]/batch)
    for i in range(batch_num):
      # get the batch data 
      batch_size = min((i+1)*batch, X.shape[0]) - i*batch
      X_batch = X[i*batch : i*batch + batch_size]
      y_batch = y[i*batch : i*batch + batch_size]
      X_batch = ndl.Tensor(X_batch, requires_grad=False)
      Z = ndl.matmul(ndl.relu(ndl.matmul(X_batch, W1)), W2)
      y_one_hot = np.eye(W2.shape[1])[y_batch]
      y_one_hot = ndl.Tensor(y_one_hot, requires_grad=False)
      loss = softmax_loss(Z, y_one_hot)
      loss.backward()
      # to numpy, then to Tensor()
      W1_data = (W1 - lr*W1.grad).numpy()
      W2_data = (W2 - lr*W2.grad).numpy()
      W1 = ndl.Tensor(W1_data)
      W2 = ndl.Tensor(W2_data)
    return W1, W2


    

    # raise NotImplementedError()
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
