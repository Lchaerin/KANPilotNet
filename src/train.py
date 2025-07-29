#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.core.protobuf import saver_pb2
from preprocess.imageSteeringDB import ImageSteeringDB
from nets.pilotNet import PilotNet
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dataset_dir', './data/datasets/driving_dataset2',
    "Directory that stores input recorded front view images and steering wheel angles.")
flags.DEFINE_bool(
    'clear_log', False,
    "Force to clear old logs if exist.")
flags.DEFINE_string(
    'train_log_dir', './logs/',
    "Directory for training logs, including summaries and checkpoints.")
flags.DEFINE_bool(
    'e', False,
    "Evaluate trained model and exit immediately.")
flags.DEFINE_float(
    'L2NormConst', 1e-3,
    "L2 regularization constant.")
flags.DEFINE_float(
    'learning_rate', 1e-4,
    "Learning rate for optimizer.")
flags.DEFINE_integer(
    'num_epochs', 30,
    "Number of epochs to train.")
flags.DEFINE_integer(
    'batch_size', 128,
    "Batch size for training.")

def train(argv=None):
    if FLAGS.e:
        print(">>> Evaluation mode: loading latest checkpoint and running on validation set.")
        with tf.Graph().as_default():
            model = PilotNet()
            dataset = ImageSteeringDB(FLAGS.dataset_dir)

            vars = tf.trainable_variables()
            loss = tf.reduce_mean(tf.square(model.y_ - model.steering)) \
                   + tf.add_n([tf.nn.l2_loss(v) for v in vars]) * FLAGS.L2NormConst

            saver = tf.train.Saver(var_list=tf.global_variables())

            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                ckpt = tf.train.latest_checkpoint(FLAGS.train_log_dir)
                if not ckpt:
                    print(f"[Error] No checkpoint found in {FLAGS.train_log_dir}")
                    return
                saver.restore(sess, ckpt)
                print(f"Restored checkpoint: {ckpt}")

                num_batches = dataset.num_images // FLAGS.batch_size
                total_loss = 0.0
                for i in range(num_batches):
                    imgs, angles = dataset.load_val_batch(FLAGS.batch_size)
                    l = sess.run(loss,
                                 feed_dict={
                                     model.image_input: imgs,
                                     model.y_: angles,
                                     model.keep_prob: 1.0
                                 })
                    total_loss += l
                avg_loss = total_loss / num_batches
                print(f"Validation batches: {num_batches}, Average loss: {avg_loss:.6f}")
        return

    if FLAGS.clear_log:
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)

    with tf.Graph().as_default():
        model = PilotNet()
        dataset = ImageSteeringDB(FLAGS.dataset_dir)

        vars = tf.trainable_variables()
        loss = tf.reduce_mean(tf.square(model.y_ - model.steering)) \
               + tf.add_n([tf.nn.l2_loss(v) for v in vars]) * FLAGS.L2NormConst
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

        # 체크포인트 세이버
        saver = tf.train.Saver(var_list=tf.global_variables())

        # 요약 정의
        tf.summary.scalar("loss", loss)
        merged_summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)
            writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)
            save_path = os.path.join(FLAGS.log_dir, "checkpoint")

            print(f"Activating TensorBoard:\n--> tensorboard --logdir={FLAGS.log_dir}\n"
                  "Connecting http://0.0.0.0:6006/")

            for epoch in range(FLAGS.num_epochs):
                num_batches = dataset.num_images // FLAGS.batch_size
                for batch in range(num_batches):
                    imgs, angles = dataset.load_train_batch(FLAGS.batch_size)
                    sess.run([loss, optimizer],
                             feed_dict={model.image_input: imgs,
                                        model.y_: angles,
                                        model.keep_prob: 0.8})

                    if batch % 10 == 0:
                        v_imgs, v_angles = dataset.load_val_batch(FLAGS.batch_size)
                        val_loss = sess.run(loss,
                                            feed_dict={model.image_input: v_imgs,
                                                       model.y_: v_angles,
                                                       model.keep_prob: 1.0})
                        print(f"Epoch {epoch}, Step {epoch*FLAGS.batch_size + batch}, Loss {val_loss:.6f}")

                    summary = sess.run(merged_summary,
                                       feed_dict={model.image_input: imgs,
                                                  model.y_: angles,
                                                  model.keep_prob: 1.0})
                    writer.add_summary(summary, epoch * num_batches + batch)
                    writer.flush()

                    if batch % FLAGS.batch_size == 0:
                        os.makedirs(save_path, exist_ok=True)
                        ckpt = os.path.join(save_path, "model.ckpt")
                        saver.save(sess, ckpt)

                print(f"Model saved: {ckpt}")
                now = datetime.now()
                logging.info(f"current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    tf.app.run(main=train)
