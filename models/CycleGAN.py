"""
Author: G. L. Roberts
Date: 25th December 2020
Description: CycleGAN implementation (small network)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

from utils.CycleGANUtils import resnet_block, generator_loss_fn,\
        discriminator_loss_fn, cycle_loss_fn


# TODO: Implement reflection padding

class CycleGAN(keras.Model):
    def __init__(self):
        super(CycleGAN, self).__init__()
        # Define generator N2D, D2N, discriminator N, D (models and optimizers)
        self.generator_D2N = get_generator('D2N')
        self.generator_N2D = get_generator('N2D')
        self.discriminator_D = get_discriminator('D')
        self.discriminator_N = get_discriminator('N')
        self.cycle_weight = tf.Variable(20.)

    def compile(self):
        super(CycleGAN, self).compile()
        self.generator_D2N_optimizer = keras.optimizers.Adam(lr=0.0002)
        self.generator_N2D_optimizer = keras.optimizers.Adam(lr=0.0002)
        self.discriminator_D_optimizer = keras.optimizers.Adam(lr=0.0002)
        self.discriminator_N_optimizer = keras.optimizers.Adam(lr=0.0002)

        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn
        self.cycle_loss_fn = cycle_loss_fn

    @tf.function
    def train_step(self, combined_input):
        night_image, day_image = combined_input
        with tf.GradientTape(persistent=True) as tape:
            # Use generators to reverse both images
            gen_day = self.generator_N2D(night_image)
            gen_night = self.generator_D2N(day_image)

            # Discriminate real and generated images
            discriminated_day = self.discriminator_D(day_image)
            discriminated_night = self.discriminator_N(night_image)
            discriminated_gen_day = self.discriminator_D(gen_day)
            discriminated_gen_night = self.discriminator_N(gen_night)

            # Discriminator loss tries to maximise probability of the
            # discriminator assigning 1s to the real image and 0s
            # to the generated image (i.e. to discriminate between the two)
            discriminator_N_loss = self.discriminator_loss_fn(
                    discriminated_night, discriminated_gen_night)
            discriminator_D_loss = self.discriminator_loss_fn(
                    discriminated_day, discriminated_gen_day)

            # Generator loss tries to maximise the probability of the
            # discriminator assigning 1s to the generated image (i.e.
            # to trick the discriminator)
            generator_D2N_loss = self.generator_loss_fn(discriminated_gen_night)
            generator_N2D_loss = self.generator_loss_fn(discriminated_gen_day)

            # Use reverse generator to get back to original images and
            # calculate a loss as the difference between the cycled
            # and true images
            cycled_night = self.generator_D2N(gen_day)
            cycled_day = self.generator_N2D(gen_night)
            cycle_loss_D2N = self.cycle_loss_fn(night_image, cycled_night,
                    tf.keras.backend.get_value(self.cycle_weight))
            cycle_loss_N2D = self.cycle_loss_fn(day_image, cycled_day,
                    tf.keras.backend.get_value(self.cycle_weight))
            cycle_loss = cycle_loss_D2N + cycle_loss_N2D

            total_generator_N2D_loss = generator_N2D_loss + cycle_loss
            total_generator_D2N_loss = generator_D2N_loss + cycle_loss

        # Calculate and apply gradients
        generator_N2D_gradients = tape.gradient(total_generator_N2D_loss,
                self.generator_N2D.trainable_variables)
        generator_D2N_gradients = tape.gradient(total_generator_D2N_loss,
                self.generator_D2N.trainable_variables)

        discriminator_N_gradients = tape.gradient(discriminator_N_loss,
                self.discriminator_N.trainable_variables)
        discriminator_D_gradients = tape.gradient(discriminator_D_loss,
                self.discriminator_D.trainable_variables)

        self.generator_N2D_optimizer.apply_gradients(
                zip(generator_N2D_gradients,
            self.generator_N2D.trainable_variables))
        self.generator_D2N_optimizer.apply_gradients(
                zip(generator_D2N_gradients,
            self.generator_D2N.trainable_variables))

        self.discriminator_N_optimizer.apply_gradients(
                zip(discriminator_N_gradients,
            self.discriminator_N.trainable_variables))
        self.discriminator_D_optimizer.apply_gradients(
                zip(discriminator_D_gradients,
            self.discriminator_D.trainable_variables))

        return {'gen_N2D_loss': total_generator_N2D_loss,
                'gen_D2N_loss': total_generator_D2N_loss,
                'disc_N_loss': discriminator_N_loss,
                'disc_D_loss': discriminator_D_loss}


def get_discriminator(name):
    input_image = keras.layers.Input(shape=(64, 64, 3))
    activation = keras.layers.LeakyReLU(alpha=0.2)
    x = layers.Conv2D(16, 4, 2, padding='same', activation=activation)(input_image)
    x = layers.Conv2D(32, 4, 2, padding='same', activation=activation)(x)
    x = layers.Conv2D(64, 4, 2, padding='same', activation=activation)(x)
    x = layers.Conv2D(1, 4, 1, padding='same', activation=activation)(x)
    return keras.models.Model(input_image, x, name=name)


def get_generator(name):
    input_image = keras.layers.Input(shape=(64, 64, 3))
    x = layers.Conv2D(16, 3, 1, padding='same', activation='relu')(input_image)
    x = layers.Conv2D(32, 3, 2, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, 2, padding='same', activation='relu')(x)
    for _ in range(6):
        resnet_layer = resnet_block((16, 16, 64))
        x = resnet_layer(x)
    x = layers.Conv2DTranspose(32, 3, 2, padding='same', output_padding=1, activation='relu')(x)
    x = layers.Conv2DTranspose(16, 3, 2, padding='same', output_padding=1, activation='relu')(x)
    x = layers.Conv2D(3, 7, 1, padding='same', activation='relu')(x)
    return keras.models.Model(input_image, x, name=name)


def main():
    pass

if __name__ == "__main__":
    main()

