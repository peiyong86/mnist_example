#!/user/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


N_CLASS = 10


def load_data():
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    assert len(train_data)==55000
    assert len(eval_data)==10000

    return train_data, train_labels, eval_data, eval_labels


def init_model():
    # create model and return
    classifier = tf.estimator.BaselineClassifier(n_classes=N_CLASS)
    return classifier


def main(argv):
    # init model
    model = init_model()
    # load data
    train_data, train_labels, eval_data, eval_labels = load_data()
    # train input fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=1,
    shuffle=True)
    # train model
    model.train(input_fn=train_input_fn)
    # eval model
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = model.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    tf.app.run()
