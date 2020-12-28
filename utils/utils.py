"""
Author: G. L. Roberts
Date: 25th December 2020
Description: Various NN building blocks
"""

import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    pass


def resnet_block(image_size):
    input_image = keras.layers.Input(shape=image_size)
    x = layers.Conv2D(64, 3, padding='same', activation=None)(input_image)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = x + input_image
    model = keras.models.Model(input_image, x)
    return model


def decode_image(fpath):
    img = tf.io.read_file(fpath)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, dtype=tf.float32) / 255
    # resize the image to the desired size
    return tf.image.resize(img, [64, 64])


def generator_loss_fn(generated):
    loss_obj = keras.losses.MeanSquaredError()
    return loss_obj(tf.ones_like(generated), generated)


def discriminator_loss_fn(real, generated):
    loss_obj = keras.losses.MeanSquaredError()
    # loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    real_loss = loss_obj(tf.ones_like(real), real)
    return 0.5 * (generated_loss + real_loss)


def cycle_loss_fn(real, cycled):
    loss = tf.reduce_mean(tf.abs(real - cycled))
    return loss * 10


def load_dataset(batch_size):
    night_ds = tf.data.Dataset.list_files("data/night/*/*.png")
    day_ds = tf.data.Dataset.list_files("data/day/*/*.png")
    night_ds = night_ds.shuffle(buffer_size=10000).map(decode_image).batch(batch_size)
    day_ds = day_ds.shuffle(buffer_size=10000).map(decode_image).batch(batch_size)
    combined_ds = tf.data.Dataset.zip((night_ds, day_ds))
    return day_ds, night_ds, combined_ds


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


class PlotExamples(keras.callbacks.Callback):
    def __init__(self, day_ds, night_ds, debugging):
        self.no_images = 4
        self.day_ds = day_ds
        self.night_ds = night_ds
        self.debugging = debugging

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(self.day_ds.take(self.no_images)):
            prediction = self.model.generator_D2N(img)[0].numpy()
            prediction = (prediction * 255).astype(np.uint8)
            img = (img[0] * 255).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = keras.preprocessing.image.array_to_img(prediction)

        if self.debugging:
            fpath = Path('images', f'DEBUG_day_combined_{epoch+1}.png')
        else:
            fpath = Path('images', f'day_combined_{epoch+1}.png')
        plt.savefig(fpath)
        plt.close()

        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(self.night_ds.take(self.no_images)):
            prediction = self.model.generator_D2N(img)[0].numpy()
            prediction = (prediction * 255).astype(np.uint8)
            img = (img[0] * 255).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = keras.preprocessing.image.array_to_img(prediction)
            # fname = f"generated_img_{i}_{epoch+1}.png"
            # prediction.save(Path('images', fname))
        if self.debugging:
            fpath = Path('images', f'DEBUG_night_combined_{epoch+1}.png')
        else:
            fpath = Path('images', f'night_combined_{epoch+1}.png')
        plt.savefig(fpath)
        plt.close()


if __name__ == "__main__":
    main()

