#!/user/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


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


if __name__ == '__main__':
    tf.app.run()
