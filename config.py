#!/usr/bin/python

# Pruning threshold setting (90 % off)
percent = 0.9

# meta path
meta_path = "./1.meta"

# model path
data_path = "./1"

# feature name(placehoder) defined in training
feature = "x"

# label name(placehoder) defined in training
label = "y_"

# define the getting data function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/', one_hot=True)
def get_data():
    batch = mnist.train.next_batch(50)
    return batch[0], batch[1]

# retrain step
retrain_step = 100
