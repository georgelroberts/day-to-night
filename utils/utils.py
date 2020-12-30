"""
Author: G. L. Roberts
Date: 30th December 2020
Description: Common utils
"""

import tensorflow as tf
import shutil
import os

def main():
    pass


def load_dataset(batch_size):
    night_ds = tf.data.Dataset.list_files("data/night/*/*.png")
    day_ds = tf.data.Dataset.list_files("data/day/*/*.png")
    night_ds = night_ds.shuffle(buffer_size=10000).map(decode_image).batch(batch_size)
    day_ds = day_ds.shuffle(buffer_size=10000).map(decode_image).batch(batch_size)
    combined_ds = tf.data.Dataset.zip((night_ds, day_ds))
    return day_ds, night_ds, combined_ds


def decode_image(fpath):
    img = tf.io.read_file(fpath)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, dtype=tf.float32) / 255
    # resize the image to the desired size
    return tf.image.resize(img, [64, 64])


def get_checkpoint_callback(run_name='checkpoint'):
    fpath = f'tmp/{run_name}/'
    if os.path.exists(fpath):
        shutil.rmtree(fpath)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
            fpath)
    return checkpoint


def get_tensorboard_callback(run_name='default_run'):
    fpath = f'logs/{run_name}/'
    if os.path.exists(fpath):
        shutil.rmtree(fpath)
    tb = tf.keras.callbacks.TensorBoard(log_dir=fpath, write_graph=False)
    return tb


if __name__ == "__main__":
    main()

