"""
Author: G. L. Roberts
Date: 30th December 2020
Description: Various utils for MUNIT
"""

# TODO: Put scales into loss functions

def main():
    pass

def reconstruction_loss_fn(x, y):
    loss_obj = keras.losses.MAE()
    return loss_obj(x, y)

def generator_loss_fn(fake):
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


if __name__ == "__main__":
    main()

