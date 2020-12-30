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


def adaptive_instance_norm(content, gamma, beta, epsilon=1e-10):
    # gamma and beta are output from MLP from style feature

    content_mean, content_var = tf.nn.moments(content, axes=[1, 2],
            keep_dims=True)
    content_std = tf.sqrt(content_var + epsilon)

    return gamma * ((content - content_mean) / content_std) + beta


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


def adaptive_resnet_block(image_size, gamma1, gamma2, beta1, beta2):
    input_image = keras.layers.Input(shape=image_size)
    x = layers.Conv2D(256, 3, padding='same', activation=None)(input_image)
    x = adaptive_instance_norm(x, gamma1, beta1)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding='same', activation=None)(x)
    x = adaptive_instance_norm(x, gamma2, beta2)
    x = x + input_image
    model = keras.models.Model(input_image, x)
    return model


if __name__ == "__main__":
    main()

