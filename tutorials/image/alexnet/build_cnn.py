"""Function to build the graph of a CNN with flexible hyperparameters
    - number of layers
    - size of the convolution kernel
    - pooling size
"""

import tensorflow as tf

def build_graph(num_layers,num_features,conv_kernel,pooling,imgsize):
    g = tf.Graph()

    with g.as_default():
        x = tf.placeholder(tf.float32, shape=[None, imgsize, imgsize, 1])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])

        inputs = x

        for i in range(num_layers):
            conv = tf.layers.conv2d(
                    inputs=inputs,
                    filters=num_features[i],
                    kernel_size=conv_kernel[i],
                    strides=[1,1],
                    padding='SAME',
                    activation=tf.nn.relu)

            pool = tf.layers.max_pooling2d(
                    inputs=conv,
                    pool_size=pooling[i],
                    strides=pooling[i],
                    padding='VALID')

            inputs = pool


        pool_flat = tf.reshape(
                tensor=pool,
                shape=[-1, (pool.shape[1]*pool.shape[2]*pool.shape[3]).value])


        dense = tf.layers.dense(
                inputs=pool_flat,
                units=1024,
                activation=tf.nn.relu)


        dropout = tf.layers.dropout(
                inputs=dense,
                rate=0.4,
                training=True)


        logits = tf.layers.dense(inputs=dropout, units=2)

        loss = tf.losses.softmax_cross_entropy(
                onehot_labels=y_, logits=logits)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return g, x, y_, train_op, accuracy, loss
