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

if __name__ == "__main__":
    main()

