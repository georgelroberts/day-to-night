"""
Author: G. L. Roberts
Date: 30th December 2020
Description: MUNIT Implementation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

from utils.MUNITUtils import reconstruction_loss_fn, generator_loss_fn,\
        discriminator_loss_fn, adaptive_instance_norm

# TODO: Implement reflection padding

class MUNIT(keras.Model):
    def __init__(self):
        super(MUNIT, self).__init__()
        # Define generator N2D, D2N, discriminator N, D (models and optimizers)
        self.style_encoder_N = get_style_encoder('style_N')
        self.style_encoder_D = get_style_encoder('style_D')
        self.content_encoder_N = get_content_encoder('content_N')
        self.content_encoder_D = get_content_encoder('content_D')
        self.decoder_N = get_decoder('decoder_N')
        self.decoder_D = get_decoder('decoder_D')
        self.discriminator_D = get_discriminator('disc_D')
        self.discriminator_N = get_discriminator('disc_N')

    def compile(self):
        super(MUNIT, self).compile()
        self.style_encoder_N_optimizer = keras.optimizers.Adam(lr=0.0001,
                beta_1=0.5, beta_2=0.999)
        self.style_encoder_D_optimizer = keras.optimizers.Adam(lr=0.0001,
                beta_1=0.5, beta_2=0.999)
        self.content_encoder_N_optimizer = keras.optimizers.Adam(lr=0.0001,
                beta_1=0.5, beta_2=0.999)
        self.content_encoder_D_optimizer = keras.optimizers.Adam(lr=0.0001,
                beta_1=0.5, beta_2=0.999)
        self.decoder_N_optimizer = keras.optimizers.Adam(lr=0.0001,
                beta_1=0.5, beta_2=0.999)
        self.decoder_D_optimizer = keras.optimizers.Adam(lr=0.0001,
                beta_1=0.5, beta_2=0.999)
        self.discriminator_D_optimizer = keras.optimizers.Adam(lr=0.0001,
                beta_1=0.5, beta_2=0.999)
        self.discriminator_N_optimizer = keras.optimizers.Adam(lr=0.0001,
                beta_1=0.5, beta_2=0.999)

        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn

    @tf.function
    def train_step(self, combined_input):
        night_image, day_image = combined_input
        with tf.GradientTape(persistent=True) as tape:
            # Get style features for both images
            style_night = self.style_encoder_N(night_image)
            style_day = self.style_encoder_D(day_image)

            # Get content features for both images
            content_night = self.content_encoder_N(night_image)
            content_day = self.content_encoder_D(day_image)

            # Reconstruct from real style and content features
            reconstructed_night_real = self.decoder_N(style_night,
                    content_night)
            reconstructed_day_real = self.decoder_D(style_day, content_day)

            # Reconstruct from real content and opposite style features
            reconstructed_night_fake = self.decoder_N(style_night,
                    content_day)
            reconstructed_day_fake = self.decoder_D(style_day, content_night)

            # Re-encode the fake images
            reencoded_content_night_fake = self.content_encoder_N(
                    reconstructed_night_fake)
            reencoded_content_day_fake = self.content_encoder_D(
                    reconstructed_day_fake)
            reencoded_style_night_fake = self.style_encoder_N(
                    reconstructed_night_fake)
            reencoded_style_day_fake = self.style_encoder_D(
                    reconstructed_day_fake)

            # Discriminator outputs
            discriminated_night_real = self.discriminator(night_image)
            discriminated_night_fake = self.discriminator(
                    reconstructed_night_real)
            discriminated_day_real = self.discriminator(day_image)
            discriminated_day_fake = self.discriminator(reconstructed_day_real)

            # Calculate losses
            L_recon_real_N = self.reconstruction_loss_fn(night_image,
                    reconstructed_night_real)
            L_recon_real_D = self.reconstruction_loss_fn(day_image,
                    reconstructed_day_real)
            L_recon_content_D = self.reconstruction_loss(content_day,
                    reencoded_content_day_fake)
            L_recon_content_N = self.reconstruction_loss(content_night,
                    reencoded_content_night_fake)
            L_recon_style_D = self.reconstruction_loss(style_day,
                    reencoded_style_day_fake)
            L_recon_style_N = self.reconstruction_loss(style_night,
                    reencoded_style_night_fake)
            L_GAN_N = self.discriminator_loss_fn(discriminated_night_real,
                    discriminated_night_fake)
            L_GAN_D = self.discriminator_loss_fn(discriminated_day_real,
                    discriminated_day_fake)

        # Calculate and apply gradients
        style_encoder_N_gradients = tape.gradient(L_recon_style_N,
                self.style_encoder_N.trainable_variables)
        style_encoder_D_gradients = tape.gradient(L_recon_style_D,
                self.style_encoder_D.trainable_variables)
        content_encoder_N_gradients = tape.gradient(L_recon_content_N,
                self.content_encoder_N.trainable_variables)
        content_encoder_D_gradients = tape.gradient(L_recon_content_D,
                self.content_encoder_D.trainable_variables)
        decoder_N_gradients = tape.gradient(
                self.decoder_N.trainable_variables)
        decoder_D_gradients = tape.gradient(
                self.decoder_D.trainable_variables)
        discriminator_N_gradients = tape.gradient(
                self.discriminator_N.trainable_variables)
        discriminator_D_gradients = tape.gradient(
                self.discriminator_D.trainable_variables)

        self.style_encoder_N_optimizer.apply_gradients(
                zip(style_encoder_N_gradients,
                    self.style_encoder_N.trainable_variables))
        self.style_encoder_D_optimizer.apply_gradients(
                zip(style_encoder_D_gradients,
                    self.style_encoder_D.trainable_variables))
        self.content_encoder_N_optimizer.apply_gradients(
                zip(content_encoder_N_gradients,
                    self.content_encoder_N.trainable_variables))
        self.content_encoder_D_optimizer.apply_gradients(
                zip(content_encoder_D_gradients,
                    self.content_encoder_D.trainable_variables))
        self.decoder_N_optimizer.apply_gradients(
                zip(decoder_N_gradients,
                    self.decoder_N.trainable_variables))
        self.decoder_D_optimizer.apply_gradients(
                zip(decoder_D_gradients,
                    self.decoder_D.trainable_variables))
        self.discriminator_N_optimizer.apply_gradients(
                zip(discriminator_N_gradients,
                    self.discriminator_N.trainable_variables))
        self.discriminator_D_optimizer.apply_gradients(
                zip(discriminator_D_gradients,
                    self.discriminator_D.trainable_variables))

        return {'encoder_N_loss': L_recon_style_N + L_recon_content_N,
                'encoder_D_loss': L_recon_style_D + L_recon_content_D,
                'decoder_N_loss': ,
                'decoder_D_loss': ,
                'discriminator_N_loss': ,
                'discriminator_D_loss':
                }


def get_discriminator(name):
    input_image = keras.layers.Input(shape=(64, 64, 3))
    activation = keras.layers.LeakyReLU(alpha=0.2)
    x = layers.Conv2D(16, 4, 2, padding='same', activation=activation)(input_image)
    x = layers.Conv2D(32, 4, 2, padding='same', activation=activation)(x)
    x = layers.Conv2D(64, 4, 2, padding='same', activation=activation)(x)
    x = layers.Conv2D(1, 4, 1, padding='same', activation=activation)(x)
    model = keras.models.Model(input_image, x, name=name)
    return model


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
    model = keras.models.Model(input_image, x, name=name)
    return model


def main():
    pass


if __name__ == "__main__":
    main()

