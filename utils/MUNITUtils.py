"""
Author: G. L. Roberts
Date: 30th December 2020
Description: Various utils for MUNIT
"""

# TODO: Put scales into loss functions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import shutil
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    pass


def reconstruction_loss_fn(x, y):
    loss_obj = keras.losses.MeanAbsoluteError()
    return loss_obj(x, y)


def generator_loss_fn(generated):
    loss_obj = keras.losses.MeanSquaredError()
    return loss_obj(tf.ones_like(generated), generated)


def discriminator_loss_fn(real, generated):
    loss_obj = keras.losses.MeanSquaredError()
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    real_loss = loss_obj(tf.ones_like(real), real)
    return generated_loss + real_loss


def adaptive_instance_norm(x, gamma, beta, epsilon=1e-10):
    # gamma and beta are output from MLP from style feature

    mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    std = tf.sqrt(var + epsilon)
    x = layers.Subtract()([x, mean])
    x = layers.Multiply()([x, 1/std])
    x = layers.Multiply()([x, gamma])
    x = layers.Add()([x, beta])
    return x


def resnet_block(image_size):
    input_image = keras.layers.Input(shape=image_size)
    x = layers.Conv2D(256, 3, padding='same', activation=None)(input_image)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding='same', activation=None)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = x + input_image
    model = keras.models.Model(input_image, x)
    return model


def adaptive_resnet_block(image_size):
    input_image = keras.layers.Input(shape=image_size)
    input_mlp = keras.layers.Input(shape=(1024))
    gamma1, gamma2, beta1, beta2 = tf.split(input_mlp, 4, axis=-1)
    x = layers.Conv2D(256, 3, padding='same', activation=None)(input_image)
    x = adaptive_instance_norm(x, gamma1, beta1)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding='same', activation=None)(x)
    x = adaptive_instance_norm(x, gamma2, beta2)
    x = x + input_image
    model = keras.models.Model([input_image, input_mlp], x)
    return model


def adain_mlp(image_size, no_params):
    input_image = keras.layers.Input(shape=image_size)
    x = layers.Dense(256)(input_image)
    x = layers.Dense(no_params)(x)
    model = keras.models.Model(input_image, x)
    return model


class PlotExamplesMUNIT(keras.callbacks.Callback):
    def __init__(self, day_ds, night_ds, debugging, folder):
        self.no_images = 4
        self.day_ds = day_ds
        self.night_ds = night_ds
        self.debugging = debugging
        self.directory = Path('images', folder)
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(self.no_images, 4, figsize=(12, 12))
        for i, (day_img, night_img) in enumerate(
                zip(self.day_ds.take(self.no_images),
                    self.night_ds.take(self.no_images))):
            style_day = self.model.style_encoder_D(day_img)
            style_night = self.model.style_encoder_N(night_img)
            content_day = self.model.content_encoder_D(day_img)
            decoded_real = self.model.decoder_D([style_day, content_day])[0].numpy()
            decoded_real = (decoded_real * 255).astype(np.uint8)
            decoded_swapped = self.model.decoder_N([style_night, content_day])[0].numpy()
            decoded_swapped = (decoded_swapped * 255).astype(np.uint8)
            day_img = (day_img[0] * 255).numpy().astype(np.uint8)
            night_img = (night_img[0] * 255).numpy().astype(np.uint8)

            ax[i, 0].imshow(day_img)
            ax[i, 1].imshow(night_img)
            ax[i, 2].imshow(decoded_real)
            ax[i, 3].imshow(decoded_swapped)
            ax[i, 0].set_title("Input day")
            ax[i, 1].set_title("Input night")
            ax[i, 2].set_title("Decoded day")
            ax[i, 3].set_title("Day to night")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
            ax[i, 2].axis("off")
            ax[i, 3].axis("off")

        if self.debugging:
            fpath = Path(self.directory, f'DEBUG_day_combined_{epoch+1}.png')
        else:
            fpath = Path(self.directory, f'day_combined_{epoch+1}.png')
        plt.savefig(fpath)
        plt.close()

        _, ax = plt.subplots(self.no_images, 4, figsize=(12, 12))
        for i, (day_img, night_img) in enumerate(
                zip(self.day_ds.take(self.no_images),
                    self.night_ds.take(self.no_images))):
            style_day = self.model.style_encoder_D(day_img)
            style_night = self.model.style_encoder_N(night_img)
            content_night = self.model.content_encoder_N(night_img)
            decoded_real = self.model.decoder_N([style_night, content_night])[0].numpy()
            decoded_real = (decoded_real * 255).astype(np.uint8)
            decoded_swapped = self.model.decoder_N([style_day, content_night])[0].numpy()
            decoded_swapped = (decoded_swapped * 255).astype(np.uint8)
            day_img = (day_img[0] * 255).numpy().astype(np.uint8)
            night_img = (night_img[0] * 255).numpy().astype(np.uint8)

            ax[i, 0].imshow(day_img)
            ax[i, 1].imshow(night_img)
            ax[i, 2].imshow(decoded_real)
            ax[i, 3].imshow(decoded_swapped)
            ax[i, 0].set_title("Input day")
            ax[i, 1].set_title("Input night")
            ax[i, 2].set_title("Decoded night")
            ax[i, 3].set_title("Night to day")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
            ax[i, 2].axis("off")
            ax[i, 3].axis("off")

        if self.debugging:
            fpath = Path(self.directory, f'DEBUG_night_combined_{epoch+1}.png')
        else:
            fpath = Path(self.directory, f'night_combined_{epoch+1}.png')
        plt.savefig(fpath)
        plt.close()


if __name__ == "__main__":
    main()

