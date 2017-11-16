import tensorflow as tf
import numpy as np


class SiCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, num_classes):
        size = 128
        self.images = tf.placeholder(tf.float32, [None, size, size, 3], name="images")
        # Define input labels and bounding boxes here...
        self.input_y = []
        self.input_bbs = []

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        batch_size = tf.shape(self.images)[0]
        image_size = tf.shape(self.images)[1]
        self.scale = tf.constant([1])
        self.square_bb = tf.tile(tf.constant([0, 0, image_size-1, image_size-1]), batch_size)
        self.blackout_bb = tf.tile([0, 0, image_size-1, image_size-1], batch_size)

        # resize to standard
        with tf.name_scope('resize') as scope:
            max_dim = tf.squeeze(tf.maximum(tf.shape(images)[1:2]))
            padded_size = tf.tile(max_dim, tf.constant([2]))
            padded_images = tf.image.resize_image_with_crop_or_pad(images, max_dim, max_dim)
            resized_images = tf.image.resize(padded_images, [size, size])

        # conv1
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(resized_images, kernel, [1, 4, 4, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)
            print_activations(conv1)
            parameters += [kernel, biases]

        # zooming1
        with tf.name_scope('zoom1') as scope:
            curr_size = tf.shape(summed_rows)[-1]
            fmap_weights = tf.Variable(tf.truncated_normal([64], dtype=tf.float32,
                                                     stddev=1e-1), name='fmap_weights')
            summed_images = tf.reduce_sum(tf.multiply(conv1, fmap_weights), axis=-1)
            summed_rows = tf.reduce_sum(summed_images, axis=1)
            summed_cols = tf.reduce_sum(summed_images, axis=2)
            linspace = tf.range(0, summed_rows.get_shape().as_list()[-1] - 1, 1)
            x_means = tf.reduce_sum(tf.multiply(summed_rows, linspace)) / tf.reduce_sum(summed_rows)
            y_means = tf.reduce_sum(tf.multiply(summed_cols, linspace)) / tf.reduce_sum(summed_cols)
            x_vars = tf.reduce_sum(tf.multiply(tf.square(linspace - x_means), summed_rows)) / tf.reduce_sum(summed_rows)
            y_vars = tf.reduce_sum(tf.multiply(tf.square(linspace - y_means), summed_cols)) / tf.reduce_sum(summed_cols)
            x_sds = tf.sqrt(x_vars)
            y_sds = tf.sqrt(y_vars)
            factor = 5
            # If proposed bounding box cuts into blackout region, reduce its size
            x_shift = (x_means + factor*x_sds) - tf.minimum(x_means + factor*x_sds, self.blackout_bb[:, 3])
            x_means -= x_shift/2
            x_sds -= x_shift/(2*factor)
            x_shift = tf.maximum(x_means - factor*x_sds, self.blackout_bb[:, 1]) - (x_means - factor*x_sds)
            x_means += x_shift/2
            x_sds -= x_shift/(2*factor)

            y_shift = (y_means + factor*y_sds) - tf.minimum(y_means + factor*y_sds, self.blackout_bb[:, 3])
            y_means -= y_shift/2
            y_sds -= y_shift/(2*factor)
            y_shift = tf.maximum(y_means - factor*y_sds, self.blackout_bb[:, 1]) - (y_means - factor*y_sds)
            y_means += y_shift/2
            y_sds -= y_shift/(2*factor)
            
            # Slicing
            max_sds = tf.maximum(x_sds, y_sds)
            y1s = y_means - factor*max_sds
            y2s = y_means + factor*max_sds
            x1s = x_means - factor*max_sds
            x2s = x_means + factor*max_sds
            square_bounding_boxes = tf.concat([y1s, x1s, y2s, x2s] + curr_size/2*factor)
            # padding so that bounding boxes remain within the limits
            padded_images = tf.pad(conv1, [[1, 1], [1, 1]] * curr_size/2*factor, "CONSTANT")
            # Crop images to bounding boxes, and zoom to current size
            cropped_images = tf.image.crop_and_resize(conv1, square_bounding_boxes, tf.range(batch_size), [curr_size, curr_size])
            # Blackout part of image corresponding to original bounding boxes (rather than the current square bounding boxes)
            # For the zoomed image, new center is just the exact center. Padding needed in original scale is (factor*x_sds).
            # Padding needed in new scale is (factor*x_sds) * [(curr_size/2) / (factor*max_sds)]
            x_means = curr_size/2
            y_means = curr_size/2
            x2s = x_means + factor*x_sds*(curr_size/2)/(factor*max_sds)
            x1s = x_means - factor*x_sds*(curr_size/2)/(factor*max_sds)
            y2s = y_means + factor*y_sds*(curr_size/2)/(factor*max_sds)
            y1s = y_means - factor*y_sds*(curr_size/2)/(factor*max_sds)
            self.blackout_bb = tf.concat([y1s, x1s, y2s, x2s])
            x_masks = tf.sequence_mask(x2s, batch_size) - tf.sequence_mask(x1s, batch_size)
            y_masks = tf.sequence_mask(y2s, batch_size) - tf.sequence_mask(y1s, batch_size)
            masks = tf.matmul(tf.reshape(y_masks, [batch_size, 1, -1]), x_masks)
            blackout_images = tf.matmul(cropped_images, masks)

            new_scale = self.scale * (curr_size/2)/(factor*max_sds)
            new_square_bb[:, 1] = self.square_bb[:, 1] + square_bounding_boxes[:, 1] / self.scale
            new_square_bb[:, 0] = self.square_bb[:, 0] + square_bounding_boxes[:, 0] / self.scale
            new_square_bb[:, 3] = self.square_bb[:, 1] + square_bounding_boxes[:, 3] / self.scale
            new_square_bb[:, 2] = self.square_bb[:, 0] + square_bounding_boxes[:, 2] / self.scale

            self.scale = new_scale
            self.square_bb = new_square_bb

        # pool1
        with tf.name_scope('pool1') as scope:
            pool1 = tf.nn.max_pool(blackout_images,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID',
                                   name='pool1')
            print_activations(pool1)
            pooling_factor = tf.shape(pool1)[1] / tf.shape(blackout_images)[1]
            self.scale = self.scale / pooling_factor
            self.blackout_bb = self.blackout_bb / pooling_factor

        # fully connected
        with tf.name_scope('fully_connected') as scope:
            flattened = tf.reshape(pool1, [batch_size, -1])
            dense = tf.layers.dense(inputs=flattened, units=384, activation=tf.nn.relu)
            dropout = tf.layers.dropout(
                inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
            logits = tf.layers.dense(inputs=dropout, units=num_classes)


    

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.predictions = tf.argmax(logits, 1, name="predictions")

        # CalculateMean cross-entropy loss # Calculate loss using logits, bounding boxes, input y.
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            l2_reg_lambda = 0
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
