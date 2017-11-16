import tensorflow as tf
import numpy as np


with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = SiCNN(num_classes=10)

    sess.run(tf.global_variables_initializer())

    def train_step(x_words_batch, x_tags_batch, x_labels_batch, x_indices_batch, x_trees_batch, y_batch, learning_rate):
        """
        A single training step
        """
        feed_dict = {
          cnn.input_words: x_words_batch,
          cnn.input_tags: x_tags_batch,
          cnn.input_labels: x_labels_batch,
          cnn.input_indices: x_indices_batch,
          cnn.input_trees: x_trees_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
          cnn.learning_rate: learning_rate
          # cnn.seq: x_words_batch.shape[1]
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}, learning_rate {:g},"
              .format(time_str, step, loss, accuracy, learning_rate))
        print(str(len(x_words_batch)) + " " + str(len(x_words_batch[0])) + " ")
        print(cnn.input_words)
        train_summary_writer.add_summary(summaries, step)

    def dev_step(x_words_batch, x_tags_batch, x_labels_batch, x_indices_batch, x_trees_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          cnn.input_words: x_words_batch,
          cnn.input_tags: x_tags_batch,
          cnn.input_labels: x_labels_batch,
          cnn.input_indices: x_indices_batch,
          cnn.input_trees: x_trees_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: 1.0
          # cnn.seq: x_words_batch.shape[1]
        }
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)

    # Generate batches
    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
    # It uses dynamic learning rate with a high value at the beginning to speed up the training
    max_learning_rate = 0.005
    min_learning_rate = 0.0001
    decay_speed = FLAGS.decay_coefficient*len(y_train)/FLAGS.batch_size
    # Training loop. For each batch...
    counter = 0
    for batch in batches:
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_speed)
        counter += 1
        x_batch, y_batch = zip(*batch)
        x_words_batch, x_tags_batch, x_labels_batch, x_indices_batch, x_trees_batch = zip(*x_batch)
        # print(np.shape(x_trees_batch))
        x_words_batch = np.array(x_words_batch)
        # print(np.shape(x_words_batch))
        x_tags_batch = np.array(x_tags_batch)
        # print(np.shape(x_tags_batch))
        x_labels_batch = np.array(x_labels_batch)
        # print(np.shape(x_labels_batch))
        x_indices_batch = np.array(x_indices_batch)
        # print(np.shape(x_indices_batch))
        x_trees_batch = list(x_trees_batch)
        x_trees_batch2 = np.zeros([x_words_batch.shape[0], x_words_batch.shape[1], x_words_batch.shape[1]])
        for i in range(len(x_trees_batch)):
            bla = eval(x_trees_batch[i])
            x_trees_batch2[i,0:len(bla),0:len(bla)] = bla
        # x_trees_batch = np.array(x_trees_batch)
        x_trees_batch = x_trees_batch2
        # print(np.shape(x_trees_batch))

        # x_trees_batch = np.reshape(x_trees_batch, (np.shape(x_words_batch)[0], np.shape(x_words_batch)[1], np.shape(x_words_batch)[1]))
        train_step(x_words_batch, x_tags_batch, x_labels_batch, x_indices_batch, x_trees_batch, y_batch, learning_rate)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            x_words_dev, x_tags_dev, x_labels_dev, x_indices_dev, x_trees_dev = zip(*x_dev)
            x_words_dev = np.array(x_words_dev)
            x_tags_dev = np.array(x_tags_dev)
            x_labels_dev = np.array(x_labels_dev)
            x_indices_dev = np.array(x_indices_dev)
            x_trees_dev = list(x_trees_dev)
            x_trees_dev2 = np.zeros([x_words_dev.shape[0], x_words_dev.shape[1], x_words_dev.shape[1]])
            for i in range(len(x_trees_dev)):
                bla = eval(x_trees_dev[i])
                x_trees_dev2[i,0:len(bla),0:len(bla)] = bla
            # x_trees_batch = np.array(x_trees_batch)
            x_trees_dev = x_trees_dev2
            # x_trees_dev = np.reshape(x_trees_dev, (np.shape(x_words_dev)[0], np.shape(x_words_dev)[1], np.shape(x_words_dev)[1]))
            dev_step(x_words_dev, x_tags_dev, x_labels_dev, x_indices_dev, x_trees_dev, y_dev, writer=dev_summary_writer)
            print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))