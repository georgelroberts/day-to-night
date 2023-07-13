"""
Author: G. L. Roberts
Date: 25th December 2020
Description: Various utils for cyclegan
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
    return keras.models.Model(input_image, x)


def generator_loss_fn(generated):
    loss_obj = keras.losses.MeanSquaredError()
    return loss_obj(tf.ones_like(generated), generated)


def discriminator_loss_fn(real, generated):
    loss_obj = keras.losses.MeanSquaredError()
    # loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    real_loss = loss_obj(tf.ones_like(real), real)
    return 0.5 * (generated_loss + real_loss)


def cycle_loss_fn(real, cycled, weight):
    loss = tf.reduce_mean(tf.abs(real - cycled))
    return loss * weight


class RecalcCycleWeight(keras.callbacks.Callback):
    def __init__(self):
        self.init_weight = 20.0

    def on_epoch_begin(self, epoch, logs=None):
        cycle_weight = self.compute_cycle_weight(epoch)
        tf.keras.backend.set_value(self.model.cycle_weight, cycle_weight)

    def compute_cycle_weight(self, epoch):
        if epoch < 0:
            return self.init_weight
        varied_weight = self.init_weight - epoch/10
        return max([0.1, varied_weight])


class PlotExamplesCycleGAN(keras.callbacks.Callback):
    def __init__(self, day_ds, night_ds, debugging, folder):
        self.no_images = 4
        self.day_ds = day_ds
        self.night_ds = night_ds
        self.debugging = debugging
        self.directory = Path('images', folder)
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(self.no_images, 2, figsize=(12, 12))
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
            fpath = Path(self.directory, f'DEBUG_day_combined_{epoch+1}.png')
        else:
            fpath = Path(self.directory, f'day_combined_{epoch+1}.png')
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
            fpath = Path(self.directory, f'DEBUG_night_combined_{epoch+1}.png')
        else:
            fpath = Path(self.directory, f'night_combined_{epoch+1}.png')
        plt.savefig(fpath)
        plt.close()


if __name__ == "__main__":
    main()

