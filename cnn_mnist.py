from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model(features, labels, mode):
    # INPUT LAYER
    input_layer = tf.reshape(features['x'], [-1,28,28,1])

    # CONVOLUTIONAL AND POOLING LAYER (1)
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pool2d(
        inputs=conv1,
        pool_size=[2,2],
        stride=2
    )

    # CONVOLUTIONAL AND POOLING LAYER (2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pool2d(
        inputs=conv2,
        pool_size=[2,2],
        stride=2
    )

    # DENSE LAYER
    pool2_flate = tf.reshape(pool2, [-1,7*7*64])
    dense = tf.layers.dense(
        inputs=pool2_flate,
        units=1024,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # LOGITS/OUTPUT LAYER
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # CALCULATE LOSS (BOTH TRAIN AND EVAL MODES)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # CONFIGURE THE TRAINING OP (FOR TRAIN MODE)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # ADD EVALUATION METRICS (FOR EVAL MODE)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classess"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


if __name__ == "__main__":
  tf.app.run()
